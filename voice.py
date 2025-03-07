import streamlit as st
import io
import wave
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder  

st.set_page_config(page_title="🎙️ Voice Debugging")
st.header("🎤 Voice Input Debugging")

# Record Audio
st.subheader("🎙️ Speak Now")
audio_dict = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹ Stop", key="mic")

if audio_dict and "bytes" in audio_dict:
    st.success("✅ Voice Recorded Successfully!")

    # Save & Display Raw Audio Data
    st.write(f"🔍 Raw Audio Size: {len(audio_dict['bytes'])} bytes")

    try:
        # Save audio to WAV
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(44100)  # Match mic sample rate
            wf.writeframes(audio_dict["bytes"])

        wav_buffer.seek(0)

        # 🎧 Playback Recorded Audio
        st.subheader("🔊 Playback Recorded Audio")
        st.audio(wav_buffer, format="audio/wav")

    except Exception as e:
        st.error(f"❌ Error saving/playing audio: {e}")
