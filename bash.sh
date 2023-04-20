#!/bin/bash
i=1
for filename in /home/nivetheni/TCI_express/out/*.jpg; do
  new_filename=$(printf "%04d.jpg" "$i")
  mv "$filename" "/home/nivetheni/TCI_express/out/$new_filename"
  let i=i+1
  let j=i*10
  if [[ ! -f "/home/nivetheni/TCI_express/out/$j.jpg" ]]; then
    i=1
  fi
done
