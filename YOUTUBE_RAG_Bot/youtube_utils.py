import re
from youtube_transcript_api import YouTubeTranscriptApi


def get_video_id(url):
    pattern = r"(?:v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})"

    match = re.search(pattern, url)

    if match:
        return match.group(1)

    return None


def get_transcript(url):
    video_id = get_video_id(url)

    if not video_id:
        raise ValueError("Invalid YouTube URL")

    api = YouTubeTranscriptApi()

    # Get all available transcripts
    transcript_list = api.list(video_id)

    # Prefer manually created English
    try:
        transcript = transcript_list.find_transcript(["en"])
        return " ".join(item.text for item in transcript.fetch())
    except Exception:
        pass

    # Try Hindi
    try:
        transcript = transcript_list.find_transcript(["hi"])
        return " ".join(item.text for item in transcript.fetch())
    except Exception:
        pass

    # Fallback to the first available transcript
    for transcript in transcript_list:
        return " ".join(item.text for item in transcript.fetch())

    raise ValueError("No transcript available for this video")