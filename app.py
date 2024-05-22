from faster_whisper import WhisperModel
import base64
import io

class InferlessPythonModel:
        
    def initialize(self):
        model_size = "large-v3"
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")

    def infer(self, inputs):
        audio_raw_ = inputs['audio']

        audio_ = base64.b64decode(audio_raw_)
        print(f'bytes received {len(audio_)}')
        buffer_ = io.BytesIO()
        buffer_.write(audio_)
        buffer_.seek(0)
        segments_, info_ = self.model.transcribe(buffer_, language='en', beam_size=5)

        segment_texts = []
        for segment_ in segments_:
            print("[%.2fs -> %.2fs] %s" % (segment_.start, segment_.end, segment_.text))
            segment_texts.append(segment_.text)

        segment = '\n'.join(segment_texts)
        return segment

    def finalize(self):
        pass
