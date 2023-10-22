from logic.valuation.freeway import FreewayValuationModule
from logic.valuation.meetingroom import MeetingRoomValuationModule

VALUATION_MODULES = {
    "freeway": FreewayValuationModule,
    "meetingroom": MeetingRoomValuationModule,
}


def get_valuation_module(env_name: str):
    if env_name not in VALUATION_MODULES.keys():
        raise NotImplementedError(f"No valuation module implemented for env '{env_name}'.")
    return VALUATION_MODULES[env_name]
