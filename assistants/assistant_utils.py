import time

from pprint import pprint
from typing import List, Dict, Any

def upload_files(clnt: object, files: List[str]) -> None:
    """
    Uploads files to the OpenAI API.
    """
    file_objs = []
    for file in files:
        with open(file, 'rb') as f:
            f_obj = clnt.files.create(file=f, purpose='assistants') 
            file_objs.append(f_obj)
    return file_objs

def print_thread_messages(clnt: object, thrd: object, content_value: bool=True) -> None:
    """
    Prints OpenAI thread messages to the console.
    """
    messages = clnt.beta.threads.messages.list(
        thread_id = thrd.id)
    for msg in messages:
        if content_value:
            pprint(msg.role + ":" + msg.content[0].text.value)
        else: 
            pprint(msg)

def loop_until_completed(clnt: object, thrd: object, run_obj: object) -> None:
    """
    Poll the Assistant runtime until the run is completed or failed
    """
    while run_obj.status not in ["completed", "failed", "requires_action"]:
        run_obj = clnt.beta.threads.runs.retrieve(
            thread_id = thrd.id,
            run_id = run_obj.id)
        time.sleep(10)
        print(run_obj.status)

def create_assistant_run(clnt: object, asst: object, thrd: object,
                         message:str) -> object:
    """
    Creates an Assistant run.
    """
    run = clnt.beta.threads.runs.create(
    thread_id=thrd.id,
    assistant_id=asst.id,
    instructions= message
)
    return run





