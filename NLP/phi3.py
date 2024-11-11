from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
import torch.quantization
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define device to use GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "microsoft/Phi-3-mini-128k-instruct"


class LM:
    """
    A language model class to load and perform inference with a causal language model.
    """

    def __init__(self, model_name, quantize=False):
        """
        Initialize the language model.

        :param model_name: Name of the pre-trained model to use.
        :param quantize: Whether to quantize the model to reduce memory usage (default is False).
        :return: None
        """
        logger.info("Initializing model called %s", model_name)
        self.config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, attn_implementation="eager"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.model.to(device)

        if quantize:
            logger.info("Quantizing the model so that it runs inference faster")
            # Quantize the model
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {
                    torch.nn.Linear
                },  # Specify the layers to quantize (e.g., Linear layers)
                dtype=torch.qint8,  # Quantization data type
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, attn_implementation="eager"
        )

    def inference(self, responses, custom_instructions="You are a helpful assistant."):
        """
        Perform inference on the input responses.

        :param responses: List of strings representing the conversation history.
        :param custom_instructions: Custom system instructions to use (default is 'You are a helpful assistant.').
        :return: The generated response text.
        """
        logger.info("Performing inference with instructions: %s", custom_instructions)
        current_role = "user"
        chat = [{"role": "system", "content": custom_instructions}]
        for res in responses:
            chat.append({"role": current_role, "content": res})
            current_role = "assistant" if current_role == "user" else "user"

        # Tokenize and perform inference
        in_ = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        out = self.model.generate(in_, max_new_tokens=3000)
        text = self.tokenizer.batch_decode(out)[0]
        return text.split("<|end|>")[-2][len("<|assistant|>") :]

    def generate_streaming(
        self,
        responses,
        custom_instructions="You are a helpful assistant.",
        continue_=True,
    ):
        """
        Generate text in a streaming fashion, token by token.

        :param responses: List of strings representing the conversation history.
        :param custom_instructions: Custom system instructions to use (default is 'You are a helpful assistant.').
        :param continue_: Whether to continue generation after a 'Continue' prompt (default is True).
        :yield: The newly generated part of the response.
        """
        logger.info("Starting streaming generation")
        current_role = "user"
        chat = [{"role": "system", "content": custom_instructions}]

        # Build chat history
        for res in responses:
            chat.append({"role": current_role, "content": res})
            current_role = "assistant" if current_role == "user" else "user"

        # Tokenize input
        in_ = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        output_tokens = in_.clone()  # Keep track of all tokens generated
        output_tokens_ = torch.tensor([]).to(device)
        while output_tokens_.dim() != output_tokens.dim():
            output_tokens_ = output_tokens_.unsqueeze(0)

        prev = ""
        while True:
            # Generate one token at a time
            outputs = self.model.generate(
                output_tokens,
                max_new_tokens=1,
                do_sample=True,  # Optional: remove for deterministic output
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Get the newly generated token and add to the sequence
            next_token = outputs[0, -1].unsqueeze(0).unsqueeze(0)
            output_tokens = torch.cat([output_tokens, next_token], dim=-1)
            output_tokens_ = torch.cat([output_tokens_, next_token], dim=-1)

            # Decode the entire sequence so far to ensure proper handling of spaces
            a = output_tokens_.squeeze().tolist()
            if isinstance(a, (int, float)):
                a = [a]

            a = [int(i) for i in a]
            generated_text = self.tokenizer.decode(a)
            generated_text = generated_text[len(prev) :]
            prev += generated_text

            # Stop if the EOS token is generated
            if "<|end|>" in generated_text:
                break

            if "Continue: " in generated_text and not continue_:
                break

            # Yield only the newly generated part
            yield generated_text


phi = LM(model_name)

if __name__ == "__main__":
    # Main program entry point
    start = input("Hello! Ask me something: ")
    r = [start]

    while True:
        total = ""
        for token in phi.generate_streaming(r):
            print(token, end="")
            total += token

        print()
        r.append(total)
        r.append(input("Continue: "))
