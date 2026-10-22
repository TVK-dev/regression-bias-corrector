try:
    # _version.py генерируется hatch-vcs при сборке/установке пакета (pip install .)
    from ._version import __version__, __version_tuple__
except ImportError:
    # Запуск напрямую из git-клона, когда src добавлен в sys.path, - файла нет.
    # Версия подтягивается из метаданных установленного дистрибутива, иначе dev0.
    try:
        from importlib.metadata import version as _get_version
        __version__ = _get_version("regression-bias-corrector")
    except Exception:
        __version__ = "0.0.0.dev0"
    _parts = __version__.split("+")[0].split(".")
    __version_tuple__ = tuple(int(p) if p.isdigit() else p for p in _parts)

version, version_tuple = __version__, __version_tuple__

from .linear_corrector import LinearBiasCorrector

__all__ = ["LinearBiasCorrector", "__version__", "__version_tuple__", "version", "version_tuple"]
