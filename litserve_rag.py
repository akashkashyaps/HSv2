import litserve as ls
from RAGFusion_Mistral import get_rag_response


class RAGAPI(ls.LitAPI):
    def setup(self, device):
        # You can initialize anything needed here, such as loading models
        pass

    def decode_request(self, request):
        # Extract the input string from the request
        return request["input"]

    def predict(self, input_string):
        # Call your get_rag_response function with the input string
        result = get_rag_response(input_string)
        yield result

    def encode_response(self, output):
        # Format the output as a dictionary
        yield {"result": output}

if __name__ == "__main__":
    api = RAGAPI()
    server = ls.LitServer(api, stream=True, accelerator="auto")
    server.run(port=8000)
