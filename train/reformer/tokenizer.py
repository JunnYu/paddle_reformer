from paddle.utils import try_import
from paddlenlp.transformers.albert.tokenizer import AlbertEnglishTokenizer


class ReformerTokenizer(AlbertEnglishTokenizer):
    resource_files_names = {
        "sentencepiece_model_file": "spiece.model",
    }
    pretrained_resource_files_map = {
        "sentencepiece_model_file": {
            "albert-base-v1": "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-base-v1.spiece.model",
            "albert-large-v1": "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-large-v1.spiece.model",
        },
    }

    pretrained_init_configuration = {
        "albert-base-v1": {"do_lower_case": True},
        "albert-large-v1": {"do_lower_case": True},
    }

    def __init__(
        self,
        sentencepiece_model_file,
        do_lower_case=True,
        remove_space=True,
        keep_accents=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.sentencepiece_model_file = sentencepiece_model_file

        spm = try_import("sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sentencepiece_model_file)
