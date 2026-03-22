# -*- coding: utf-8 -*-
project = 'Triton 中文文档'
copyright = '2024, Triton 社区（中文翻译）'
author = 'Triton 社区'
release = 'main'
language = 'zh_CN'

extensions = [
    'sphinx.ext.mathjax',
    'myst_parser',
]

templates_path = []
exclude_patterns = ['_build']

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
}
html_title = 'Triton 中文文档'
html_logo = 'https://cdn.openai.com/triton/assets/triton-logo.png'
html_show_sourcelink = False
html_extra_path = []

source_suffix = {'.rst': 'restructuredtext'}
master_doc = 'index'
