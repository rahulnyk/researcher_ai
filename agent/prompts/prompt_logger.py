from ..agent_logger import AgentLogger
from yachalk import chalk

al = AgentLogger(name="PROMPT LOG")
prompt_logger = al.getLogger()

def log_prompt(message: str):
    prompt_logger.critical(chalk.gray(message))


