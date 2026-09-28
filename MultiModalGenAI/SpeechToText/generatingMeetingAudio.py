from gtts import gTTS

def convert_to_speech(story, filename="generated_transcript.mp3"):

    tts = gTTS(
        text=story,
        lang="en"
    )

    tts.save(filename)

    return filename

if __name__ == "__main__":

    story = """Priya: Good morning everyone. Thanks for joining the Tech Infra tower sync. Today we'll cover network, cloud and storage, security, and open risks. Rahul, let's start with you.

Rahul: Thanks Priya. The data center switch upgrade is at seventy percent. Two of the four core switches are migrated with zero downtime. The remaining two are scheduled for this Saturday night, in the approved maintenance window.

Priya: Any risks there?

Rahul: One. The firmware for the third switch arrived late from the vendor. If it fails validation, we slip by a week. I'll confirm by Thursday.

Priya: Okay, please flag it early. Anita, cloud and storage.

Anita: Sure. The cloud cost optimization is on track. We right-sized about forty virtual machines, and that's saving roughly twelve percent on monthly spend. On storage, the backup array is at eighty-five percent capacity, which is above our threshold.

Priya: That's a concern. What's the plan?

Anita: We've raised a purchase request for additional disks. Meanwhile, I'll archive older snapshots to cheaper storage, which should free around fifteen percent by next week.

Priya: Good. Please share the request number so I can push procurement. Vikram, security.

Vikram: Yes. The quarterly patching cycle is complete for ninety-two percent of servers. The remaining eight percent are legacy systems that need application owner approval. I've escalated those twice already.

Rahul: I can help. Some of those sit behind my network segment, so I can add temporary access controls until they're patched.

Vikram: That would really help. Also, the multi-factor authentication rollout for admin accounts is finished, and we've closed all critical audit findings from last month.

Priya: Excellent work. Let me summarize the action items. Rahul confirms switch firmware by Thursday. Anita archives old snapshots and shares the purchase request number. Vikram and Rahul coordinate temporary controls for legacy servers. And I'll follow up with application owners on approvals.

Anita: Sounds good. Can we also have a short review next Monday to check progress?

Priya: Yes, I'll send an invite. Anything else? No? Great, thanks everyone. Have a productive week.

Rahul: Thanks, bye.

Vikram: Thanks all.

Anita: Bye everyone."""

    print("\nConverting story to speech...")

    filename = convert_to_speech(story)

    print(f"\nAudio saved as: {filename}")
    print("Your personal storyteller is ready!")