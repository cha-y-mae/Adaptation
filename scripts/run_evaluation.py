'''

This script is for running an evaluation experiment defined by a yaml config: loads data, queries a model handler, saves predictions. It saves partial results periodically and evaluates partial outputs on interruption, final outputs on success. 

'''

import json
import logging
import atexit
import signal
from pathlib import Path

from tqdm import tqdm
from scripts.utils import load_config, save_predictions
from models import load_model_handler
from evaluations.evaluator import evaluate, split_prediction

SAVE_EVERY = 50


def run_experiment(config_path):
    logging.info(f"loading config from {config_path}")
    config = load_config(config_path)

    task_type = config["task"]["type"]
    if task_type not in ("mcq", "answer_generation"):
        raise ValueError(f"unsupported task.type:{task_type} expected mcq or answer_generation")

    logging.info(f"initializing model:{config['model']['name']}")
    model_handler = load_model_handler(config)

    logging.info(f"loading dataset from {config['dataset']['path']}")
    dataset_path = config["dataset"]["path"]
    instruction_path = config["dataset"]["instruction_path"]

    model_cfg = config.get("model", {})
    batch_size = model_cfg.get("batch_size", 4)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(instruction_path, "r", encoding="utf-8") as f:
        instruction = f.read().strip()

    output_path = config["output"]["predictions_path"]
    metrics_path = config["output"]["metrics_path"]
    partial_path = str(Path(output_path).with_suffix(".partial.csv"))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

    predictions = []
    finished = False

    def save_now(path):
        save_predictions(predictions, path)

    def finalize_and_eval(path_for_eval):
        if predictions:
            save_now(path_for_eval)
        try:
            m = evaluate(path_for_eval, metrics_path, task_type)
            logging.info(f"[partial] metrics saved to {metrics_path}")
            logging.info(f"[partial] {json.dumps(m, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logging.warning(f"[partial] evaluation failed:{e}")

    def on_exit():
        if not finished:
            finalize_and_eval(partial_path)

    atexit.register(on_exit)

    def handle_sigint(sig, frame):
        logging.info("caught ctrl+c saving and evaluating partial results")
        finalize_and_eval(partial_path)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    use_batch = hasattr(model_handler, "prompt_batch")

    if task_type == "mcq":
        max_tokens = model_cfg.get("max_tokens", 8)
    else:
        max_tokens = model_cfg.get("max_tokens", 256)

    if use_batch:
        logging.info(f"using prompt_batch batch_size={batch_size}")

        for start in tqdm(range(0, len(dataset), batch_size), desc="processing examples"):
            batch_items = dataset[start : start + batch_size]

            batch_input_texts = []
            batch_meta = []

            for offset, item in enumerate(batch_items):
                global_idx = start + offset + 1
                input_text = item.get("Question")
                if not input_text:
                    logging.warning(f"no input text found for item {global_idx} skipping")
                    continue
                batch_input_texts.append(input_text)
                batch_meta.append((global_idx, item))

            if not batch_input_texts:
                continue

            batch_predictions = model_handler.prompt_batch(
                batch_input_texts,
                instruction=instruction,
                task_type=task_type,
                max_tokens=max_tokens,
            )

            for (global_idx, item), prediction in zip(batch_meta, batch_predictions):
                letter_gt = item.get("Answer")
                text_gt = item.get("answer_text")

                gt_for_eval = letter_gt if task_type == "mcq" else text_gt
                pred_main, pred_aux = split_prediction(prediction, task_type)

                pred_expl = None
                pred_conf = None
                if task_type == "answer_generation":
                    pred_expl = pred_aux

                predictions.append(
                    {
                        "id": global_idx,
                        "input": item.get("Question"),
                        "prediction": prediction,
                        "prediction_letter": pred_main if task_type == "mcq" else None,
                        "prediction_explanation": pred_expl,
                        "prediction_confidence": pred_conf,
                        "ground_truth": gt_for_eval,
                        "ground_truth_letter": letter_gt,
                        "ground_truth_text": text_gt,
                    }
                )

                if len(predictions) % SAVE_EVERY == 0:
                    save_now(partial_path)
                    logging.info(f"saved {len(predictions)} partial predictions to {partial_path}")

    else:
        logging.info("using prompt per-example")

        for idx, item in enumerate(tqdm(dataset, desc="processing examples"), start=1):
            input_text = item.get("Question")
            if not input_text:
                logging.warning(f"no input text found for item {idx} skipping")
                continue

            prediction = model_handler.prompt(
                input_text,
                instruction,
                task_type=task_type,
                max_tokens=max_tokens,
            )

            letter_gt = item.get("Answer")
            text_gt = item.get("answer_text")

            gt_for_eval = letter_gt if task_type == "mcq" else text_gt
            pred_main, pred_aux = split_prediction(prediction, task_type)

            if idx <= 3:
                logging.info(f"[debug] raw prediction:{repr(prediction)[:500]}")
                logging.info(f"[debug] split main={pred_main} aux={repr(pred_aux)[:200]}")

            pred_expl = None
            pred_conf = None
            if task_type == "answer_generation":
                pred_expl = pred_aux

            predictions.append(
                {
                    "id": idx,
                    "input": input_text,
                    "prediction": prediction,
                    "prediction_letter": pred_main if task_type == "mcq" else None,
                    "prediction_explanation": pred_expl,
                    "prediction_confidence": pred_conf,
                    "ground_truth": gt_for_eval,
                    "ground_truth_letter": letter_gt,
                    "ground_truth_text": text_gt,
                }
            )

            if len(predictions) % SAVE_EVERY == 0:
                save_now(partial_path)
                logging.info(f"saved {len(predictions)} partial predictions to {partial_path}")

    logging.info(f"saving predictions to {output_path}")
    save_now(output_path)
    save_now(partial_path)

    logging.info("starting evaluation")
    m = evaluate(output_path, metrics_path, task_type)
    logging.info(f"evaluation completed metrics saved to {metrics_path}")
    logging.info(f"metrics:{json.dumps(m, indent=4, ensure_ascii=False)}")

    finished = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")
    import sys

    config_file = sys.argv[1]
    try:
        run_experiment(config_file)
    except Exception as e:
        logging.error(f"an error occurred:{e}")
        raise
