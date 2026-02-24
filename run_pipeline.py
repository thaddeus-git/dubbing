#!/usr/bin/env python3
"""
Run VideoLingo pipeline programmatically for demo.mkv
Converts English video to Chinese with Qwen-TTS dubbing
"""

# NOTE: We use local_files_only=True for WhisperX to use cached models
# We allow Qwen TTS to download from HF since it's not cached
import os
import sys
import shutil

# Set up path - src contains the core module
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, current_dir)
os.environ['PATH'] += os.pathsep + current_dir

# Video file
VIDEO_FILE = 'output/demo.mkv'


def main():
    print('='*70)
    print('VideoLingo Pipeline: English → Chinese with Qwen-TTS')
    print('='*70)
    print(f'Input: {VIDEO_FILE}')
    print()

    # Verify input exists
    if not os.path.exists(VIDEO_FILE):
        print(f'Error: {VIDEO_FILE} not found!')
        return False

    # Clean up any previous output
    output_dir = 'output'
    if os.path.exists(os.path.join(output_dir, 'output_sub.mp4')):
        print('Cleaning up previous output...')
        for f in ['output_sub.mp4', 'output_dub.mp4']:
            path = os.path.join(output_dir, f)
            if os.path.exists(path):
                os.remove(path)
                print(f'  Removed {f}')

    print()
    print('Starting 12-step pipeline...')
    print('-'*70)

    try:
        # Step 1: Download video (skip - we have local file)
        print('[Step 1/12] Skip - Local video file')

        # Step 2: WhisperX ASR transcription
        print('[Step 2/12] WhisperX ASR - English transcription with word timestamps...')
        from core.step2_whisperX import transcribe
        transcribe()
        print('  ✓ Transcription complete')

        # Step 3a: spaCy sentence splitting
        print('[Step 3/12] Sentence splitting (spaCy + LLM)...')
        from core.step3_1_spacy_split import split_by_spacy
        split_by_spacy()
        print('  ✓ spaCy splitting complete')

        from core.step3_2_splitbymeaning import split_sentences_by_meaning
        split_sentences_by_meaning()
        print('  ✓ LLM splitting complete')

        # Step 4: Summary and translation
        print('[Step 4/12] Summary generation and translation to Chinese...')
        from core.step4_1_summarize import get_summary
        get_summary()
        print('  ✓ Summary complete')

        from core.step4_2_translate_all import translate_all
        translate_all()
        print('  ✓ Translation complete')

        # Step 5: Subtitle splitting
        print('[Step 5/12] Subtitle splitting and alignment...')
        from core.step5_splitforsub import split_for_sub_main
        split_for_sub_main()
        print('  ✓ Subtitle splitting complete')

        # Step 6: Timeline generation
        print('[Step 6/12] Timeline generation...')
        from core.step6_generate_final_timeline import align_timestamp_main
        align_timestamp_main()
        print('  ✓ Timeline generation complete')

        # Step 7: Merge subtitles to video
        print('[Step 7/12] Merging subtitles to video...')
        from core.step7_merge_sub_to_vid import merge_subtitles_to_video
        merge_subtitles_to_video()
        print('  ✓ Subtitles merged')

        # Step 8: Audio task generation
        print('[Step 8/12] Audio task generation...')
        from core.step8_1_gen_audio_task import gen_audio_task_main
        gen_audio_task_main()
        print('  ✓ Audio tasks generated')

        from core.step8_2_gen_dub_chunks import gen_dub_chunks
        gen_dub_chunks()
        print('  ✓ Dub chunks generated')

        # Step 9: Extract reference audio
        print('[Step 9/12] Extract reference audio...')
        from core.step9_extract_refer_audio import extract_refer_audio_main
        extract_refer_audio_main()
        print('  ✓ Reference audio extracted')

        # Step 10: Generate audio with Qwen-TTS
        print('[Step 10/12] Generate TTS audio (Qwen-TTS for Chinese)...')
        from core.step10_gen_audio import gen_audio
        gen_audio()
        print('  ✓ Audio generation complete')

        # Step 11: Merge full audio
        print('[Step 11/12] Merge full audio with speed adjustment...')
        from core.step11_merge_full_audio import merge_full_audio
        merge_full_audio()
        print('  ✓ Audio merged')

        # Step 12: Merge dub to video
        print('[Step 12/12] Merge final dub to video...')
        from core.step12_merge_dub_to_vid import merge_video_audio
        merge_video_audio()
        print('  ✓ Final video complete')

    except Exception as e:
        print(f'\n✗ Error at step: {e}')
        import traceback
        traceback.print_exc()
        return False

    print()
    print('='*70)
    print('Pipeline Complete!')
    print('='*70)
    print()
    print('Output files:')
    sub_video = os.path.join(output_dir, 'output_sub.mp4')
    dub_video = os.path.join(output_dir, 'output_dub.mp4')

    if os.path.exists(sub_video):
        size_mb = os.path.getsize(sub_video) / (1024*1024)
        print(f'  ✓ Subtitled video: {sub_video} ({size_mb:.1f} MB)')
    if os.path.exists(dub_video):
        size_mb = os.path.getsize(dub_video) / (1024*1024)
        print(f'  ✓ Dubbed video: {dub_video} ({size_mb:.1f} MB)')

    print()
    print('To play the dubbed video:')
    print(f'  ffplay {dub_video}')
    print(f'  # or')
    print(f'  open {dub_video}')

    return True


if __name__ == '__main__':
    main()
