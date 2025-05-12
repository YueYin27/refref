from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from r3f_ns.refref_dataparser import RefRefDataParserConfig


refref_dataparser = DataParserSpecification(
    config=RefRefDataParserConfig(),
    description="Data parser for RefRef dataset",
)