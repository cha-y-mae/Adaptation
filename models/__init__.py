from .openai_handler import OpenAIHandler
# do same here to import other model handlers 

import os  # To access environment variables

def load_model_handler(config):
    '''
    loading the appropriate model handler based on config.
    '''
    model_type = config["model"]["type"]
    matching_mode = config.get("task", {}).get("matching", "letter")

    if model_type == "openai":
        #retrieve API key from environment
        api_key = config["model"].get("api_key", os.getenv("OPENAI_API_KEY"))
        if not api_key:
            raise ValueError(
                "API key is not provided in config or environment variable (OPENAI_API_KEY)."
            )
        return OpenAIHandler(
            api_key=api_key,
            model=config["model"]["name"]
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
