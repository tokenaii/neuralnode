# Replica Voice IDs

NeuralNode Replica exposes **20 curated custom voice IDs** for users.

- Use the `replic-*` ID in your code.
- NeuralNode resolves it internally to the original `edge_tts` voice.
- Legacy `edge_tts` IDs still resolve for compatibility, but the custom Replica IDs are the supported public interface.

## Usage

```python
from neuralnode.replica import ReplicaTTS

tts = ReplicaTTS(voice_id="replic-salma-language{ar-eg}")
tts.speak("مرحبا من نيرال نود")
```

## Voice Table

| Custom Replica Voice ID | Original edge_tts Voice ID | Name | Language | Gender | Description |
| --- | --- | --- | --- | --- | --- |
| `replic-aria-language{en-us}` | `en-US-AriaNeural` | Aria | `en-us` | female | American English female - professional |
| `replic-guy-language{en-us}` | `en-US-GuyNeural` | Guy | `en-us` | male | American English male - professional |
| `replic-jenny-language{en-us}` | `en-US-JennyNeural` | Jenny | `en-us` | female | American English female - friendly |
| `replic-sonia-language{en-gb}` | `en-GB-SoniaNeural` | Sonia | `en-gb` | female | British English female |
| `replic-ryan-language{en-gb}` | `en-GB-RyanNeural` | Ryan | `en-gb` | male | British English male |
| `replic-salma-language{ar-eg}` | `ar-EG-SalmaNeural` | Salma | `ar-eg` | female | Arabic Egyptian female |
| `replic-shakir-language{ar-eg}` | `ar-EG-ShakirNeural` | Shakir | `ar-eg` | male | Arabic Egyptian male |
| `replic-zariyah-language{ar-sa}` | `ar-SA-ZariyahNeural` | Zariyah | `ar-sa` | female | Arabic Saudi female |
| `replic-denise-language{fr-fr}` | `fr-FR-DeniseNeural` | Denise | `fr-fr` | female | French female |
| `replic-henri-language{fr-fr}` | `fr-FR-HenriNeural` | Henri | `fr-fr` | male | French male |
| `replic-elvira-language{es-es}` | `es-ES-ElviraNeural` | Elvira | `es-es` | female | Spanish female |
| `replic-dalia-language{es-mx}` | `es-MX-DaliaNeural` | Dalia | `es-mx` | female | Mexican Spanish female |
| `replic-katja-language{de-de}` | `de-DE-KatjaNeural` | Katja | `de-de` | female | German female |
| `replic-conrad-language{de-de}` | `de-DE-ConradNeural` | Conrad | `de-de` | male | German male |
| `replic-elsa-language{it-it}` | `it-IT-ElsaNeural` | Elsa | `it-it` | female | Italian female |
| `replic-francisca-language{pt-br}` | `pt-BR-FranciscaNeural` | Francisca | `pt-br` | female | Brazilian Portuguese female |
| `replic-nanami-language{ja-jp}` | `ja-JP-NanamiNeural` | Nanami | `ja-jp` | female | Japanese female |
| `replic-xiaoxiao-language{zh-cn}` | `zh-CN-XiaoxiaoNeural` | Xiaoxiao | `zh-cn` | female | Chinese (Simplified) female |
| `replic-sunhi-language{ko-kr}` | `ko-KR-SunHiNeural` | SunHi | `ko-kr` | female | Korean female |
| `replic-emel-language{tr-tr}` | `tr-TR-EmelNeural` | Emel | `tr-tr` | female | Turkish female |
