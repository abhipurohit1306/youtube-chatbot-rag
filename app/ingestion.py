from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled
)

from langchain_core.documents import Document


def get_video_chunks(video_id):

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id,languages=["en"])
        documents = []
        current_text = []

        current_start = None
        current_end = None
        target_chunk_size = 1000

        for chunk in transcript_list:
            if current_start is None:
                current_start = chunk.start

            current_text.append(
                chunk.text
            )

            current_end = (chunk.start +chunk.duration)

            combined_text = " ".join(current_text)

            if len(combined_text) >= target_chunk_size:

                documents.append(
                    Document(
                        page_content=combined_text,
                        metadata={
                            "start": current_start,
                            "end": current_end
                        }
                    )
                )

                current_text = []
                current_start = None
                current_end = None

        # leftover text

        if current_text:

            documents.append(
                Document(
                    page_content=" ".join(
                        current_text
                    ),
                    metadata={
                        "start": current_start,
                        "end": current_end
                    }
                )
            )

        return documents

    except TranscriptsDisabled:

        print(
            "Transcripts are disabled for this video."
        )

        return []