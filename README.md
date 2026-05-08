# Ai Builder : 💩Poop Classification💩

## Link:
- [Click for Streamlit](https://chokunclassification.streamlit.app/)
- [Click for Medium](https://medium.com/@aeeseedee7788/poop-classification-a92033bfe255)
- [Train Models Code](https://github.com/chokun7788/PoopforAIB/blob/main/Chokun7788.ipynb)

## Overview:
This project is a **student project** developed as part of a research assignment. It is **not** a medical tool and should **not** be used for any medical diagnosis or treatment purposes. This project is intended solely for research and educational purposes.

## License:
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.  
It is provided for **research** purposes only, and the use of this software for commercial purposes is prohibited.

### Important Disclaimer:
This project is developed for research purposes only. It is not intended to be used as a medical tool and should not be relied upon for any health or medical-related decision-making. Use it at your own discretion.

## Deploy to Streamlit
1. Push the repository to GitHub.
2. In Streamlit Cloud, connect this GitHub repo.
3. Set the app entrypoint to `Prototype/app.py`.
4. If the model file is not stored in the repo, set a secret or environment variable `MODEL_URL` to a public download URL for `convnextv2_thev1_best_for_good.pkl`.
5. Do not commit the `.pkl` model file directly if it exceeds GitHub file-size limits.

## Notes
- The app now supports loading the model from a local file or from the URL configured in `MODEL_URL`.
- The local model path is `convnextv2_thev1_best_for_good.pkl` at the project root.
- The `.gitignore` file excludes the local model file and cache folder to avoid accidental commits.
