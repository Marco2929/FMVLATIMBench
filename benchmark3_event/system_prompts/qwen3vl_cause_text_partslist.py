from . import parts

SYSTEM_PROMPT = f"""You are analyzing "The Incredible Machines 2". 
Based on the image, provide a direct answer to the question in TASK_DESCRIPTION. 

## Parts in the game
{parts.PARTS_LIST}
"""
