import os
from datetime import datetime


def findFile(dir, strings=None, fileExtensions=False, isFolder=False):
    """
    Finds all the files in 'dir' that contain one string from 'strings' in their name.
    Additional parameters:
        'fileExtensions': True/False : Look for a specific file extension
        'isFloder': look for folders
    """
    filesInDir = []
    foundFiles = []
    cntFound = 0
    # Find files or subdir in dir
    for filename in os.listdir(dir):
        if isFolder:
            if os.path.isdir(os.path.join(dir, filename).replace("\\", "/")):
                filesInDir.append(os.path.join(dir, filename).replace("\\", "/"))
        else:
            if os.path.isfile(os.path.join(dir, filename).replace("\\", "/")):
                filesInDir.append(os.path.join(dir, filename).replace("\\", "/"))

    # Find files that contain the keyword
    if filesInDir:
        for file in filesInDir:
            if not isFolder:
                # Define what is to be searched in
                filename, extension = os.path.splitext(file)
                if fileExtensions:
                    fileText = extension
                else:
                    fileText = os.path.basename(filename)
            else:
                fileText = os.path.basename(file)
            # Check for translations
            if isinstance(strings, list):
                for string in strings:
                    if string in fileText:
                        foundFiles.append(file)
                        cntFound += 1
            elif isinstance(strings, str):
                if strings in fileText:
                    foundFiles.append(file)
                    cntFound += 1
            else:
                foundFiles.append(file)
                cntFound += 1

    sortedFiles = sorted(foundFiles)
    if cntFound == 1:
        sortedFiles = sortedFiles[0]
    return sortedFiles, cntFound


def moveFile(files, origin, destination, tracker):
    """
    Move all the files in 'files' from 'origin' to 'destination'.
    Additional parameters:
        'tracker': True/False : enable file tracking
    """
    if not os.path.exists(destination):
        os.mkdir(destination)
        print("destination dir has been created")

    if tracker:
        trackername = "filetracker.txt"
        if not os.path.isfile(os.path.join(origin, trackername).replace("\\", "/")):
            trackerorigin = open(
                os.path.join(origin, trackername).replace("\\", "/"), "a"
            )
            print("tracker file has been created in the origin directory")
            trackerorigin.close()
        if not os.path.isfile(
            os.path.join(destination, trackername).replace("\\", "/")
        ):
            trackerdestination = open(
                os.path.join(destination, trackername).replace("\\", "/"), "a"
            )
            print("tracker file has been created in the destination directory")
            trackerdestination.close()

    for file in files:
        if os.path.isfile(os.path.join(origin, file).replace("\\", "/")):
            now = datetime.now()  # current date and time to be used as a tag
            log = (
                now.strftime("< %m/%d/%Y, %H:%M:%S > : ")
                + "move <"
                + file
                + "> from <"
                + origin
                + "> to <"
                + destination
                + ">"
            )
            os.rename(
                os.path.join(origin, file).replace("\\", "/"),
                os.path.join(destination, file).replace("\\", "/"),
            )
            if tracker:
                trackerorigin = open(
                    os.path.join(origin, trackername).replace("\\", "/"), "a"
                )
                trackerorigin.write(log + "\n")
                trackerorigin.close()
                trackerdestination = open(
                    os.path.join(destination, trackername).replace("\\", "/"), "a"
                )
                trackerdestination.write(log + "\n")
                trackerdestination.close()


def renameFile(dir, oldname, newname):
    """
    For a list of files in 'dir' change their 'oldname' in 'newname'.
    """
    namelists = [oldname, newname]
    check = len(namelists[0])
    for l in namelists:
        if check != len(l):
            raise ValueError("lists of names have different lengths!")
    # check for size concistency
    for old, new in zip(oldname, newname):
        os.rename(
            os.path.join(dir, old).replace("\\", "/"),
            os.path.join(dir, new).replace("\\", "/"),
        )


def add_prefix(dir, names, prefix):
    """
    For a list of files in 'dir' add prefix to names.
    """
    for old in names:
        new = prefix + old
        os.rename(
            os.path.join(dir, old).replace("\\", "/"),
            os.path.join(dir, new).replace("\\", "/"),
        )


def add_suffix(dir, names, suffix):
    """
    For a list of files in 'dir' add suffix to names.
    """
    for old in names:
        new = old + suffix
        os.rename(
            os.path.join(dir, old).replace("\\", "/"),
            os.path.join(dir, new).replace("\\", "/"),
        )
