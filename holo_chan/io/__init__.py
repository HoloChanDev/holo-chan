from typing import TYPE_CHECKING

from holo_chan.io.interfaces import InputSource, OutputSink

if TYPE_CHECKING:
    from holo_chan.io.input_local import LocalInputSource
    from holo_chan.io.output_local import LocalSpeaker

__all__ = [
    "InputSource",
    "OutputSink",
]
