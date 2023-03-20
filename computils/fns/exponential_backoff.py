import random
import time


def exponential_backoff(
    target,
    target_args=(),
    target_kwargs={},
    retry_exception=Exception,
    scale=5.0,
    base=2.0,
    max_retries=5,
    max_delay=60.0,
    jitter=0.1,
):
    """Exponential backoff with jitter.

    :param scale: Scale of the exponential backoff.
    :param base: Base of the exponential backoff.
    :param max_retries: Maximum number of retries.
    :param max_delay: Maximum delay.
    :param jitter: Jitter factor.
    :return: Generator of delays.
    """
    for i in range(max_retries):
        delay = min(scale * (base**i), max_delay)
        delay *= 1 + jitter * random.uniform(-1, 1)

        try:
            return target(*target_args, **target_kwargs)
        except retry_exception as e:
            if i == max_retries - 1:
                raise e
            else:
                time.sleep(delay)
    raise RuntimeError("Exponential backoff failed after {max_retries} retries.")
