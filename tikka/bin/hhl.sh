#! /bin/bash

JARS=`echo ./lib/*.jar | tr ' ' ':'`
CP=build/classes:$JARS

# java -classpath $CP -Xmx512m BadMetaphor.apps.Train -e m1 -i 10 -d /Users/tsmoon/Corpora/psychtoy -c conll2k -r 2 > hhl.m1.psychtoy.out

#java -classpath $CP -Xmx512m BadMetaphor.apps.Train -e m1 -i 10  -d /Users/tsmoon/Corpora/psych -c conll2k -r 2 > hhl.m1.psychtoy.out

# java -classpath $CP -Xmx512m tikka.apps.Train -e m1 -i 10 -pi 10 -pr 0.8 -pt 0.1 -d /Users/tsmoon/Corpora/psychreview_from_topic_toolbox -c conll2k -r 2 > hhl.m1.psych.out

# java -classpath $CP -Xmx512m tikka.apps.Train -e m1 -itr 10 -ks 10 -kl 1 \
#     -d /Users/tsmoon/Corpora/psychtoy -c conll2k -r 2 -ot out/hhl.m1.psychtoy.tab.out \
#     -os out/hhl.m1.psychtoy.sample.out -n out/m1.psychtoy \
#     -m models/hhl.m1.psychtoy.model

# java -classpath $CP -Xmx512m tikka.apps.Tagger -e m1 -ks 10 -kl 1 \
#     -d /Users/tsmoon/Corpora/psychtoy -c conll2k -r 2 \
#     -os out/hhl.m1.psychtoy.sample.out  \
#     -l models/hhl.m1.psychtoy.model

java -classpath $CP -Xmx512m tikka.apps.Tagger -e m1 -c conll2k \
    -l models/hhl.m1.psychtoy.model -ks 5 -kl 1 -os sample.out \
    -j out/test -f /Users/tsmoon/Corpora/tikka_data/psychreview_from_topic_toolbox/dev
