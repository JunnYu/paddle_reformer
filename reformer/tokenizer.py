from paddle.utils import try_import
from paddlenlp.transformers.albert.tokenizer import AlbertEnglishTokenizer


class ReformerTokenizer(AlbertEnglishTokenizer):
    resource_files_names = {
        "sentencepiece_model_file": "spiece.model",
    }
    pretrained_resource_files_map = {
        "sentencepiece_model_file": {
            "reformer-crime-and-punishment": "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/spiece.model",
        },
    }

    pretrained_init_configuration = {
        "reformer-crime-and-punishment": {"do_lower_case": False},
    }

    def __init__(
        self,
        sentencepiece_model_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        eos_token="</s>",
        unk_token="<unk>",
        **kwargs
    ):

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.sentencepiece_model_file = sentencepiece_model_file

        spm = try_import("sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sentencepiece_model_file)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return len(token_ids_0) * [0] + len(token_ids_1 ) * [1]