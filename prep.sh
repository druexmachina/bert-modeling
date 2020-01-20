#!/bin/bash

# Usage: bert_prep.sh input_file [output_file_prefix]
#
# Takes a .csv file with two columns (label and data), then removes
# the header row and separates the columns into separate files

if [[ $# -eq 0 ]] ; then
    echo 'No input file provided'
    exit 0
else
    if [[ $# -eq 1 ]] ; then
        label=label.txt
        data=data.txt
    else
        if [[ $# -eq 2 ]] ; then
            label=$2_label.txt
            data=$2_data.txt
        else
            echo 'Invalid number of arguments provided'
            exit 0
        fi
    fi
    sed 's|^\([a-zA-z]*\),\(.*\)|\1|' $1 > $label
    sed -e 's|^\([a-zA-z]*\),\(.*\)|\2|' -e 's|"\(.*\)"|\1|' $1 > $data
    echo -e 'Does the file have a header? (y/n): '
    read header
    case $header in
        'y' | 'Y')
            # File has header; trim first line
            sed -i '1d' $label
            sed -i '1d' $data
            ;;
        *)
            ;;
    esac
fi


