from urllib.parse import urlparse, parse_qs


def extract_video_id(url):

    parsed_url = urlparse(url)

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    if parsed_url.hostname in (
        "www.youtube.com",
        "youtube.com"
    ):
        return parse_qs(
            parsed_url.query
        ).get("v", [None])[0]

    return None


def format_timestamp(seconds):

    total_seconds = int(seconds)

    hours = total_seconds // 3600

    minutes = (total_seconds % 3600) // 60

    seconds = total_seconds % 60

    if hours > 0:

        return f"{hours:02}:{minutes:02}:{seconds:02}"

    return f"{minutes:02}:{seconds:02}"

def is_small_talk(question):

    small_talk = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "how are you"
    }

    return question.lower().strip() in small_talk