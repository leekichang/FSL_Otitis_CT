$crop = @('112', '144')

foreach ( $c in $crop ){
    python ./src/train.py --crop $c
}