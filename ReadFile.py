__author__ = 'carsten'

import csv
import zipfile
import logging
import os.path
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def read_file(filename):
    return pd.read_csv(filename)


@DeprecationWarning
def __read_file(filename, zipped=False, head=True, max_lines=None):
    entries = None
    if zipped:
        if zipfile.is_zipfile(filename):
            z = zipfile.ZipFile(filename)
            if len(z.filelist) == 1:
                csv_file = z.open(z.filelist[0])
                entries = _read_csv(csv_file, head, max_lines)
            else:
                logger.error(
                    "Invalid input file (File {0} is not a Zip file containing exactly one file.)".format(filename))
            z.close()
        else:
            logger.error("'{0}' is not a Zip file.".format(filename))
    else:
        if os.path.exists(filename):
            entries = _read_csv(open(filename), head, max_lines)
        else:
            logger.error("File '{0}' does not exist.".format(filename))
    return entries


@DeprecationWarning
def _read_csv(csv_file, head=True, max_lines=None):
    logger.info("Reading {1} lines from '{0}'...".format(csv_file, "all" if max_lines is None else max_lines))
    entries = list()
    csv_reader = csv.reader(csv_file)
    if head:
        logger.debug("Skipping head line.")
        csv_reader.next()

    while max_lines is None or len(entries) < max_lines:
        try:
            entries.append(csv_reader.next())
        except StopIteration:
            break
    logger.info("{1} entries read from '{0}'.".format(csv_file, len(entries)))
    return entries

