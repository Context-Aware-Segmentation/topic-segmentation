# Adapted from fastNLP (Apache-2.0) for PhenoBERT; see THIRD_PARTY_NOTICES.md.
"""
.. todo::
    doc
"""

__all__ = [
    "cached_path",
    "get_filepath",
    "get_cache_path",
    "split_filename_suffix",
    "get_from_cache",
]

import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests import HTTPError
from tqdm import tqdm

from ..core import logger

PRETRAINED_BERT_MODEL_DIR = {
    'en': 'bert-base-cased.zip',
    'en-large-cased-wwm': 'bert-large-cased-wwm.zip',
    'en-large-uncased-wwm': 'bert-large-uncased-wwm.zip',

    'en-large-uncased': 'bert-large-uncased.zip',
    'en-large-cased': 'bert-large-cased.zip',

    'en-base-uncased': 'bert-base-uncased.zip',
    'en-base-cased': 'bert-base-cased.zip',

    'en-base-cased-mrpc': 'bert-base-cased-finetuned-mrpc.zip',

    'en-distilbert-base-uncased': 'distilbert-base-uncased.zip',

    'multi-base-cased': 'bert-base-multilingual-cased.zip',
    'multi-base-uncased': 'bert-base-multilingual-uncased.zip',

    'cn': 'bert-chinese-wwm.zip',
    'cn-base': 'bert-base-chinese.zip',
    'cn-wwm': 'bert-chinese-wwm.zip',
    'cn-wwm-ext': "bert-chinese-wwm-ext.zip"
}

PRETRAINED_ELMO_MODEL_DIR = {
    'en': 'elmo_en_Medium.zip',
    'en-small': "elmo_en_Small.zip",
    'en-original-5.5b': 'elmo_en_Original_5.5B.zip',
    'en-original': 'elmo_en_Original.zip',
    'en-medium': 'elmo_en_Medium.zip'
}

PRETRAIN_STATIC_FILES = {
    'en': 'glove.840B.300d.zip',

    'en-glove-6b-50d': 'glove.6B.50d.zip',
    'en-glove-6b-100d': 'glove.6B.100d.zip',
    'en-glove-6b-200d': 'glove.6B.200d.zip',
    'en-glove-6b-300d': 'glove.6B.300d.zip',
    'en-glove-42b-300d': 'glove.42B.300d.zip',
    'en-glove-840b-300d': 'glove.840B.300d.zip',
    'en-glove-twitter-27b-25d': 'glove.twitter.27B.25d.zip',
    'en-glove-twitter-27b-50d': 'glove.twitter.27B.50d.zip',
    'en-glove-twitter-27b-100d': 'glove.twitter.27B.100d.zip',
    'en-glove-twitter-27b-200d': 'glove.twitter.27B.200d.zip',

    'en-word2vec-300': "GoogleNews-vectors-negative300.txt.gz",

    'en-fasttext-wiki': "wiki-news-300d-1M.vec.zip",
    'en-fasttext-crawl': "crawl-300d-2M.vec.zip",

    'cn': "tencent_cn.zip",
    'cn-tencent': "tencent_cn.zip",
    'cn-fasttext': "cc.zh.300.vec.gz",
    'cn-sgns-literature-word': 'sgns.literature.word.txt.zip',
    'cn-char-fastnlp-100d': "cn_char_fastnlp_100d.zip",
    'cn-bi-fastnlp-100d': "cn_bi_fastnlp_100d.zip",
    "cn-tri-fastnlp-100d": "cn_tri_fastnlp_100d.zip"
}

DATASET_DIR = {
    # Classification, English
    'aclImdb': "imdb.zip",
    "yelp-review-full": "yelp_review_full.tar.gz",
    "yelp-review-polarity": "yelp_review_polarity.tar.gz",
    "sst-2": "SST-2.zip",
    "sst": "SST.zip",

    # Classification, Chinese
    "chn-senti-corp": "chn_senti_corp.zip",
    "weibo-senti-100k": "WeiboSenti100k.zip",
    "thuc-news": "THUCNews.zip",

    # Matching, English
    "mnli": "MNLI.zip",
    "snli": "SNLI.zip",
    "qnli": "QNLI.zip",
    "rte": "RTE.zip",

    # Matching, Chinese
    "cn-xnli": "XNLI.zip",

    # Sequence Labeling, Chinese
    "msra-ner": "MSRA_NER.zip",
    "peopledaily": "peopledaily.zip",
    "weibo-ner": "weibo_NER.zip",

    # Chinese Word Segmentation
    "cws-pku": 'cws_pku.zip',
    "cws-cityu": "cws_cityu.zip",
    "cws-as": 'cws_as.zip',
    "cws-msra": 'cws_msra.zip',

    # Summarization, English
    "ext-cnndm": "ext-cnndm.zip",

    # Question & answer
    "cmrc2018": "cmrc2018.zip"

}

PRETRAIN_MAP = {'elmo': PRETRAINED_ELMO_MODEL_DIR,
                "bert": PRETRAINED_BERT_MODEL_DIR,
                "static": PRETRAIN_STATIC_FILES}

FASTNLP_EXTEND_DATASET_URL = 'fastnlp_dataset_url.txt'
FASTNLP_EXTEND_EMBEDDING_URL = {'elmo': 'fastnlp_elmo_url.txt',
                         'bert':'fastnlp_bert_url.txt',
                         'static': 'fastnlp_static_url.txt'
}


def cached_path(url_or_filename: str, cache_dir: str = None, name=None) -> Path:
    if cache_dir is None:
        data_cache = Path(get_cache_path())
    else:
        data_cache = cache_dir

    if name:
        data_cache = os.path.join(data_cache, name)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, Path(data_cache))
    elif parsed.scheme == "" and Path(os.path.join(data_cache, url_or_filename)).exists():
        # File, and it exists.
        return Path(os.path.join(data_cache, url_or_filename))
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found in {}.".format(url_or_filename, data_cache))
    else:
        # Something unknown
        raise ValueError(
            "unable to parse {} as a URL or as a local path".format(url_or_filename)
        )


def get_filepath(filepath):
    if os.path.isdir(filepath):
        files = os.listdir(filepath)
        if len(files) == 1:
            return os.path.join(filepath, files[0])
        else:
            return filepath
    elif os.path.isfile(filepath):
        return filepath
    else:
        raise FileNotFoundError(f"{filepath} is not a valid file or directory.")


def get_cache_path():
    if 'FASTNLP_CACHE_DIR' in os.environ:
        fastnlp_cache_dir = os.environ.get('FASTNLP_CACHE_DIR')
        if os.path.isdir(fastnlp_cache_dir):
            return fastnlp_cache_dir
        else:
            raise NotADirectoryError(f"{os.environ['FASTNLP_CACHE_DIR']} is not a directory.")
    fastnlp_cache_dir = os.path.expanduser(os.path.join("~", ".fastNLP"))
    return fastnlp_cache_dir


def _get_base_url(name):
    environ_name = "FASTNLP_{}_URL".format(name.upper())

    if environ_name in os.environ:
        url = os.environ[environ_name]
        if url.endswith('/'):
            return url
        else:
            return url + '/'
    else:
        URLS = {
            'embedding': "http://212.129.155.247/embedding/",
            "dataset": "http://212.129.155.247/dataset/"
        }
        if name.lower() not in URLS:
            raise KeyError(f"{name} is not recognized.")
        return URLS[name.lower()]


def _get_embedding_url(embed_type, name):
    _filename = FASTNLP_EXTEND_EMBEDDING_URL.get(embed_type, None)
    if _filename:
        url = _read_extend_url_file(_filename, name)
        if url:
            return url
    embed_map = PRETRAIN_MAP.get(embed_type, None)
    if embed_map:
        filename = embed_map.get(name, None)
        if filename:
            url = _get_base_url('embedding') + filename
            return url
        raise KeyError("There is no {}. Only supports {}.".format(name, list(embed_map.keys())))
    else:
        raise KeyError(f"There is no {embed_type}. Only supports bert, elmo, static")

def _read_extend_url_file(filename, name)->str:
    cache_dir = get_cache_path()
    filepath = os.path.join(cache_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        if name == parts[0]:
                            return parts[1]
    return None

def _get_dataset_url(name):
    url = _read_extend_url_file(FASTNLP_EXTEND_DATASET_URL, name)
    if url:
        return url

    filename = DATASET_DIR.get(name, None)
    if filename:
        url = _get_base_url('dataset') + filename
        return url
    else:
        raise KeyError(f"There is no {name}.")


def split_filename_suffix(filepath):
    filename = os.path.basename(filepath)
    if filename.endswith('.tar.gz'):
        return filename[:-7], '.tar.gz'
    return os.path.splitext(filename)


def get_from_cache(url: str, cache_dir: Path = None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = re.sub(r".+/", "", url)
    dir_name, suffix = split_filename_suffix(filename)

    match_dir_name = match_file(dir_name, cache_dir)
    if match_dir_name:
        dir_name = match_dir_name
    cache_path = cache_dir / dir_name

    # get cache path to put the file
    if cache_path.exists():
        return get_filepath(cache_path)

    # response = requests.head(url, headers={"User-Agent": "fastNLP"})
    # if response.status_code != 200:
    #     raise IOError(
    #         f"HEAD request failed for url {url} with status code {response.status_code}."
    #     )

    # add ETag to filename if it exists
    # etag = response.headers.get("ETag")

    if not cache_path.exists():
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        # GET file object
        req = requests.get(url, stream=True, headers={"User-Agent": "fastNLP"})
        if req.status_code == 200:
            success = False
            fd, temp_filename = tempfile.mkstemp()
            uncompress_temp_dir = None
            try:
                content_length = req.headers.get("Content-Length")
                total = int(content_length) if content_length is not None else None
                progress = tqdm(unit="B", total=total, unit_scale=1)
                logger.info("%s not found in cache, downloading to %s" % (url, temp_filename))

                with open(temp_filename, "wb") as temp_file:
                    for chunk in req.iter_content(chunk_size=1024 * 16):
                        if chunk:  # filter out keep-alive new chunks
                            progress.update(len(chunk))
                            temp_file.write(chunk)
                progress.close()
                logger.info(f"Finish download from {url}")

                if suffix in ('.zip', '.tar.gz', '.gz'):
                    uncompress_temp_dir = tempfile.mkdtemp()
                    logger.debug(f"Start to uncompress file to {uncompress_temp_dir}")
                    if suffix == '.zip':
                        unzip_file(Path(temp_filename), Path(uncompress_temp_dir))
                    elif suffix == '.gz':
                        ungzip_file(temp_filename, uncompress_temp_dir, dir_name)
                    else:
                        untar_gz_file(Path(temp_filename), Path(uncompress_temp_dir))
                    filenames = os.listdir(uncompress_temp_dir)
                    if len(filenames) == 1:
                        if os.path.isdir(os.path.join(uncompress_temp_dir, filenames[0])):
                            uncompress_temp_dir = os.path.join(uncompress_temp_dir, filenames[0])

                    cache_path.mkdir(parents=True, exist_ok=True)
                    logger.debug("Finish un-compressing file.")
                else:
                    uncompress_temp_dir = temp_filename
                    cache_path = str(cache_path) + suffix

                logger.info(f"Copy file to {cache_path}")
                if os.path.isdir(uncompress_temp_dir):
                    for filename in os.listdir(uncompress_temp_dir):
                        if os.path.isdir(os.path.join(uncompress_temp_dir, filename)):
                            shutil.copytree(os.path.join(uncompress_temp_dir, filename), cache_path / filename)
                        else:
                            shutil.copyfile(os.path.join(uncompress_temp_dir, filename), cache_path / filename)
                else:
                    shutil.copyfile(uncompress_temp_dir, cache_path)
                success = True
            except Exception as e:
                logger.error(e)
                raise e
            finally:
                if not success:
                    if cache_path.exists():
                        if cache_path.is_file():
                            os.remove(cache_path)
                        else:
                            shutil.rmtree(cache_path)
                os.close(fd)
                os.remove(temp_filename)
                if os.path.isdir(uncompress_temp_dir):
                    shutil.rmtree(uncompress_temp_dir)
                elif os.path.isfile(uncompress_temp_dir):
                    os.remove(uncompress_temp_dir)
            return get_filepath(cache_path)
        else:
            raise HTTPError(f"Status code:{req.status_code}. Fail to download from {url}.")


def unzip_file(file: Path, to: Path):
    # unpack and write out in CoNLL column-like format
    from zipfile import ZipFile

    with ZipFile(file, "r") as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(to)


def untar_gz_file(file: Path, to: Path):
    import tarfile

    with tarfile.open(file, 'r:gz') as tar:
        tar.extractall(to)


def ungzip_file(file: str, to: str, filename:str):
    import gzip

    g_file = gzip.GzipFile(file)
    with open(os.path.join(to, filename), 'wb+') as f:
        f.write(g_file.read())
    g_file.close()


def match_file(dir_name: str, cache_dir: Path) -> str:
    files = os.listdir(cache_dir)
    matched_filenames = []
    for file_name in files:
        if re.match(dir_name + '$', file_name) or re.match(dir_name + '\\..*', file_name):
            matched_filenames.append(file_name)
    if len(matched_filenames) == 0:
        return ''
    elif len(matched_filenames) == 1:
        return matched_filenames[-1]
    else:
        raise RuntimeError(f"Duplicate matched files:{matched_filenames}, this should be caused by a bug.")
