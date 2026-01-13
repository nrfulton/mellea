from mellea.stdlib.base import Context, ModelOutputThunk, Component, CBlock
from mellea.stdlib.sampling import SamplingStrategy
from mellea.stdlib.chat import Message
import json
from typing import Generic, Type, Any, Literal
from mellea.backends.formatter import TemplateRepresentation
import typing_extensions

class JSONDiff(Component):
    ...

class Upsert(JSONDiff):
    def __init__(self, changed_kvs: dict):
        self.changed_kvs = changed_kvs
    
    def format_for_llm(self):
        return json.dumps(self.changed_kvs, indent=2)


class MessageWithDiff(Component[list[str | JSONDiff]]):
    def __init__(self, role: str, content: str, diffs: list[JSONDiff]):
        self.role = role
        self.content = content
        self._diffs = diffs

    def parts(self):
        return self._diffs

    def format_for_llm(self):
        return TemplateRepresentation(
            obj=self,
            role=self.role,
            args={
                "diffs": self._diffs,
                "content": self.content,
            },
        )

    def _parse(self, computed: ModelOutputThunk[Any]) -> "MessageWithDiff":
        value: str = computed.value
        assert value is not None, "library error: should never call parse() on an uncomputed mot"
        parsed_parts: list[str | JSONDiff] = list()
        curr_idx = 0
        curr_part : str | None = None
        curr_part_type: Literal['diff', 'text'] = 'text'
        while curr_idx < len(value):
            match curr_part_type:
                case 'text':
                    if value[curr_idx:curr_idx+len("```json")] == "```json":
                        if curr_part is not None and curr_part.strip().lstrip() != "":
                            parsed_parts.append(curr_part)
                        curr_part = ""
                        curr_part_type = "diff"
                        curr_idx += len("```json")
                    else:
                        if curr_part is None:
                            curr_part = ""
                        curr_part += value[curr_idx]
                        curr_idx += 1
                case 'diff':
                    assert curr_part is not None
                    if value[curr_idx:curr_idx+3] == "```":
                        try:
                            parsed_diff = json.loads(curr_part)
                        except Exception as e:
                            raise TypeError(f"Expected diff {curr_part} to be valid JSON but got error during parsing: {e}")
                        parsed_parts.append(Upsert(parsed_diff))
                        curr_part_type = 'text'
                        curr_part = None
                        curr_idx += len("```")
                    else:
                        curr_part += value[curr_idx]
                        curr_idx += 1
                case _:
                    raise TypeError()
        text_parts = [part for part in parsed_parts if type(part) is str]
        diff_parts = [part for part in parsed_parts if part not in text_parts]
        assert all([type(p) != str for p in diff_parts]), f"Found strings in diff_parts: {diff_parts}; text_parts was: {text_parts} with diff types: {[type(part) for part in diff_parts]}"
        return MessageWithDiff(role="assistant", content="".join(text_parts), diffs=diff_parts)


class ArtifactChatContext(Context):
    """An ArtifactChatContext is a chat about an artifact that evolves over time.
    
    TODO give examples and explain how a chat would work."""

    def __init__(self, chat_history: list[MessageWithDiff | ModelOutputThunk], origin_artifact: dict | list):
        self._chat_history = chat_history
        self._origin_artifact = origin_artifact
        self._system_msg = Message(role="system", content="You are an assistant whose job is to make changes to a JSON object. If the user asks a question about the JSON, answer their question. If the user asks you to make changes to the JSON, reply with a JSON object containing keys and their new values. For example, if the user says 'change the value of x to 5 and change th value of y to 2', then you should reply with ```json\n{\"x\": 5,\n\"y\": 2}\n```")
    
    def update_artifact(self, artifact: dict | list, diff: JSONDiff) -> dict | list:
        new_artifact = artifact.copy()
        match diff:
            case Upsert():
                for key, value in diff.changed_kvs.items():
                    new_artifact[key] = value
            case _:
                raise Exception(f"Diff type not supported: {type(diff)} for diff {diff}")
        return new_artifact

    def _process_msg_diffs(self, artifact: list | dict, msg: MessageWithDiff | ModelOutputThunk[MessageWithDiff]) -> list | dict:
        match msg:
            case MessageWithDiff():
                new_artifact = artifact.copy()
                for diff in msg._diffs:
                    new_artifact = self.update_artifact(new_artifact, diff)
                return new_artifact
            case ModelOutputThunk():
                assert msg.is_computed(), "Object Protocol Error in the Artifact-based chat library: must wait for all actions in the context to be computed before asking for the current state of the artifact."
                assert msg.parsed_repr is not None, "Core error: his assert should NEVER fail because we already asserted is_computed"
                return self._process_msg_diffs(artifact, msg.parsed_repr)
            case _:
                raise TypeError(f"Expected MessageWithDiff or ModelOutputThunk[MessageWithDiff] but found {type(msg)}")

    def get_artifact(self):
        artifact = self._origin_artifact
        for msg in self._chat_history:
            artifact = self._process_msg_diffs(artifact, msg)
        return artifact
    
    def add(self, mot: ModelOutputThunk[MessageWithDiff]) -> "ArtifactChatContext":
        new_history = self._chat_history.copy()
        new_history.append(mot)
        return ArtifactChatContext(chat_history=new_history, origin_artifact=self._origin_artifact)

    def view_for_generation(self) -> list[CBlock | Component]:
        view = [self._system_msg]
        view.extend(self._chat_history)
        view.append(Message(role="user", content=f"The document is: \n```json\n{json.dumps(self.get_artifact(), indent=4)}\n```"))
        return view


async def main():
    import mellea
    m = mellea.start_session()
    m.chat('what is 1+1?')
    backend = m.backend
    ctx = ArtifactChatContext(chat_history=[], origin_artifact={})
    mot, new_ctx = await backend.generate_from_context(MessageWithDiff(role="user", content="Set x to 2 in the dictionary.", diffs=[]), ctx=ctx)
    await mot.avalue()
    result = new_ctx.get_artifact() # no type hints here because backend.generate_from_context doesn't know that context types are preserved.
    print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())