# Third-party notices

harnesslag itself is MIT licensed (see [LICENSE](LICENSE)). It uses or ships the following.

| What | Where | Source | License |
|---|---|---|---|
| GSM8K (train and test splits) | downloaded to `data/` at run time, not committed | https://github.com/openai/grade-school-math | MIT |
| SmolLM2-135M-Instruct | downloaded from the Hugging Face Hub at run time | https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct | Apache-2.0 |
| Qwen2.5-0.5B-Instruct (GPU script only, not run) | downloaded at run time | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct | Apache-2.0 |
| Albrecht Dürer, *Knight, Death and the Devil* (1513), National Gallery of Art 1943.3.3519 | `docs/art/durer_knight_death_devil.jpg`, banner left edge (dithered) | https://commons.wikimedia.org/wiki/File:Albrecht_D%C3%BCrer,_Knight,_Death_and_Devil,_1513,_NGA_6637.jpg | Public domain / CC0 (NGA Open Access) |
| Albrecht Dürer, *The Four Horsemen of the Apocalypse* (c. 1498) | `docs/art/durer_four_riders.jpg`, banner right edge (dithered, mirrored) | https://commons.wikimedia.org/wiki/File:Durer_Revelation_Four_Riders.jpg | Public domain |
| UnifrakturMaguntia (banner title, rendered as a bitmap) | `docs/art/UnifrakturMaguntia-Book.ttf` | https://github.com/google/fonts/tree/main/ofl/unifrakturmaguntia | SIL Open Font License 1.1 |
| DejaVu Sans Mono (banner tagline, GIF text) | system font, not shipped | https://dejavu-fonts.github.io | Bitstream Vera / DejaVu license |

Python dependencies (PyTorch, Transformers, NumPy, SciPy, Matplotlib, Pillow) are installed from PyPI
under their own licenses and are not vendored.
