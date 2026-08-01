"""Canonical UNSB model name for target-reference style conditioning."""

from .dosc_sb_model import DoscSBModel


class TrscSBModel(DoscSBModel):
    """Compatibility-safe canonical alias for the historical DOSC model."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = DoscSBModel.modify_commandline_options(
            parser,
            is_train=is_train,
        )
        parser.set_defaults(dataset_mode="trsc_unaligned")
        return parser
