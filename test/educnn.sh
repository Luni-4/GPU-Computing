#!/bin/sh

educnn_path="../sequential/educnn/build/sources/"

rm -f test_educnn.txt

echo "Configurazione 1" > test_educnn.txt
$educnn_path./test_configurazione1 >> test_educnn.txt

echo "Configurazione 2" >> test_educnn.txt
$educnn_path./test_configurazione2 >> test_educnn.txt

echo "Configurazione 3" >> test_educnn.txt
$educnn_path./test_configurazione3 >> test_educnn.txt
