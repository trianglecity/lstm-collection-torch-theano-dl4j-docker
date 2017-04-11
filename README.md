##
## The Makefile is based on https://docs.docker.com/opensource/project/set-up-dev-env/
## 


This is a collection of LSTM examples from torch-autograd, theano and deeplearning4j.


NOTICE1: the penn-tree-bank is used for the torch-autograd LSTM.

NOTICE2-1: The Large Movie Review Dataset (aclimdb_v1.tar.gz) is used for the theano LSTM.
NOTICE2-2: imdb_homemade.pkl is used instead of imdb.pkl
NOTICE2-3: Python code for imdb_homemade.pkl is in lstm_examples/theano-lstm/movie_review/(bow_homemade.py).

NOTICE3: The Complete Works of William Shakespeare (by World Library) is used for the dl4j LSTM.

NOTICE4: Do NOT change pom.xml files in the deeplearning4j.


Please follow the instructions below to run the LSMT examples.


[1] download (or git clone) this source code folder

[2] cd downloaded-source-code-folder

[3] sudo make BIND_DIR=. shell

	wait ... wait ... then a bash shell will be ready (root@937ac3c8f4f4:/#)

[4 torch-autograd instructions]

[4-1] root@937ac3c8f4f4:/# cd /root/lstm/lua

[4-2] root@937ac3c8f4f4:~/lstm/lua# cd torch

[4-3] root@937ac3c8f4f4:~/lstm/lua/torch# ./clean.sh

[4-4] root@937ac3c8f4f4:~/lstm/lua/torch# TORCH_LUA_VERSION=LUA53 ./install.sh

[4-5] type in yes [and enter].

[4-6] root@937ac3c8f4f4:~/lstm/lua/torch# source /root/.bashrc

[4-7] root@937ac3c8f4f4:~/lstm/lua/torch# cd ..

[4-8] root@937ac3c8f4f4:~/lstm/lua# cd torch-autograd/

[4-9] root@937ac3c8f4f4:~/lstm/lua/torch-autograd# luarocks make

[4-10] root@937ac3c8f4f4:~/lstm/lua/torch-autograd# cd ..

[4-11] root@937ac3c8f4f4:~/lstm/lua# cd ..

[4-12] root@937ac3c8f4f4:~/lstm# cd lstm_examples/

[4-13] root@937ac3c8f4f4:~/lstm/lstm_examples# cd torch-autograd-lstm/

[4-14] root@937ac3c8f4f4:~/lstm/lstm_examples/torch-autograd-lstm# lua ./train-penn-rnn.lua

[4-15] the output may look like


	Loaded datasets: 	table: 0x18a54c0
	to lstm opt.wordDim = 	200
	to lstm opt.hiddens = 	200
	nClasses = 	10000
	 torch.nDimension(words) = 	2
	 words:size(1) = 	10000
	 words:size(2) = 	200
	torch.nDimension(trainData) = 	1
	trainData:size(1) = 	929589
	epochLength = 	92958
	torch.nDimension(trainData) = 	2
	trainData:size(1) = 	10
	trainData:size(2) = 	92958

	Training Epoch #1
	in epoch x:size(1) = ...10...... 1/92958 ...........]  ETA: 0ms | Step: 0ms
	in epoch x:size(2) = 	20
	in epoch y:size(1) = 	10
	in epoch y:size(2) = 	20
	n_params = 	4
	in f, 	table
	in f, 	table
	in f, 	table
	in f, 	nil
	in epoch f, batchSize = 	10
	in epoch f, bpropLength = 	20
	in epoch f, nElements = 	200
	in epoch f torch.nDimension(params.words.W) = 	2
	in epoch f, torch.size(params.words.W, 1) = 	10000
	in epoch f, torch.size(params.words.W, 2) = 	200
	in f torch.nDimension(h2) = 	3
	in f h2:size(1) = 	10
	in f h2:size(2) = 	20
	in f h2:size(3) = 	200
	in f torch.nDimension(yf) = 	1
	in f yf:size(1) = 	200
	in epoch x:size(1) = ...10...... 21/92958 ........]  ETA: 7h11m | Step: 278ms
	in epoch x:size(2) = 	20
	in epoch y:size(1) = 	10
	in epoch y:size(2) = 	20
	n_params = 	4
	in f, 	table
	in f, 	table
	in f, 	table
	in f, 	nil
	in epoch f, batchSize = 	10
	in epoch f, bpropLength = 	20
	in epoch f, nElements = 	200
	in epoch f torch.nDimension(params.words.W) = 	2
	in epoch f, torch.size(params.words.W, 1) = 	10000
	in epoch f, torch.size(params.words.W, 2) = 	200
	in f torch.nDimension(h2) = 	3
	in f h2:size(1) = 	10
	in f h2:size(2) = 	20
	in f h2:size(3) = 	200
	in f torch.nDimension(yf) = 	1
	in f yf:size(1) = 	200
	...
	...


[5 theano instructions] 

[5-1] root@806dcd2bf8fc:/# cd /root/lstm/python

[5-1] root@806dcd2bf8fc:~/lstm/python# cd Theano/

[5-2] root@806dcd2bf8fc:~/lstm/python/Theano# python setup.py develop

[5-3] root@806dcd2bf8fc:~/lstm/python/Theano# cd ..

[5-4] root@806dcd2bf8fc:~/lstm/python# cd ..

[5-5] root@806dcd2bf8fc:~/lstm# cd lstm_examples/

[5-6] root@806dcd2bf8fc:~/lstm/lstm_examples# cd theano-lstm/

[5-7] the output may look like

	
	... get dataset ...
	Loading data
	data_dir = 
	data_file =  imdb_homemade.pkl
	Building model ...
	model is ready ..., but wait for a while

	in lstm_layer
		state_below dim 1 = max_length_in_the_minibatch
		state_below dim 2 =  16
		state_below dim 3 =  128
	in scan.py, x is already a list
	in scan.py, x is already a list
	 x is None
	in scan.py, n_seqs =  2
	in scan.py, (n_seqs) i =  0
		adding a seqs to OrderedDict
		dictionary
	in scan.py, (n_seqs) i =  1
		adding a seqs to OrderedDict
		dictionary
	in scan.py, mintap =  0
	in scan.py, maxtap =  0
	in scan.py, for k in seq['taps'], k =  0
	in scan.py, actual_slice =  Subtensor{int64}.0
	in scan.py, nw_slice.name =  None
	in scan.py, [scan_seqs] addpending nw_seq
	in scan.py, [inner_seqs] addpending nw_slice
	in scan.py, len(scan_seqs) =  1
	in scan.py, mintap =  0
	in scan.py, maxtap =  0
	in scan.py, for k in seq['taps'], k =  0
	in scan.py, actual_slice =  Subtensor{int64}.0
	in scan.py, [scan_seqs] addpending nw_seq
	in scan.py, [inner_seqs] addpending nw_slice
	in scan.py, len(scan_seqs) =  2
	in scan.py, seq.shape[0] =  Subtensor{int64}.0
	in scan.py, seq.shape[0] =  Subtensor{int64}.0
	in scan.py, n_steps =  Subtensor{int64}.0
	in scan.py, zip_counter =  0
		in scan.py, nw_seq.name =  mask[0:]
	in scan.py, zip_counter =  1
		in scan.py, getattr =  None
	in scan.py, x.type() =  <TensorType(float64, matrix)>
	in scan.py, x.type() =  <TensorType(float64, 3D)>
	in scan.py, final n_seqs =  2
	in scan.py, scan_op.Scan(inner_inputs, new_outs, info)
	in scan.py, actual_n_steps =  Subtensor{int64}.0
	Optimization
	16008 train examples
	843 valid examples
	500 test examples
	Epoch  0 Update  10 Cost  0.694935843509
	Epoch  0 Update  20 Cost  0.691621212642
	Epoch  0 Update  30 Cost  0.693230048931
	Epoch  0 Update  40 Cost  0.692581579123
	Epoch  0 Update  50 Cost  0.695363084298
	Epoch  0 Update  60 Cost  0.688050757456
	Epoch  0 Update  70 Cost  0.690628624395
	Epoch  0 Update  80 Cost  0.696064907199
	Epoch  0 Update  90 Cost  0.705444248737
	Epoch  0 Update  100 Cost  0.690083960852
	...


[6 dl4j instructions]


	This source code includes 
	javacpp
	javacpp-presets (for OpenBLAS)
	libnd4j
	nd4j
	datavec
	deeplearning4j
	exmaples (GravesLSTMCharModellingExample java example code)


[6-1] root@7778850ee17b:/# cd /root/lstm/java

[6-2] root@7778850ee17b:~/lstm/java#cd dl4j

[6-3] root@7778850ee17b:~/lstm/java/dl4j# cd javacpp

[6-4] root@7778850ee17b:~/lstm/java/dl4j/javacpp# mvn clean install

[6-5] root@7778850ee17b:~/lstm/java/dl4j/javacpp# cd ..

[6-6] root@7778850ee17b:~/lstm/java/dl4j# cd javacpp-presets

[6-7] root@7778850ee17b:~/lstm/java/dl4j/javacpp-presets# ./cppbuild.sh

[6-8] root@7778850ee17b:~/lstm/java/dl4j/javacpp-presets# mvn clean install

[6-9] root@7778850ee17b:~/lstm/java/dl4j/javacpp-presets# cd ..

[6-10] root@7778850ee17b:~/lstm/java/dl4j# cd libnd4j/

[6-11] root@7778850ee17b:~/lstm/java/dl4j/libnd4j#  ./buildnativeoperations.sh

[6-12] root@7778850ee17b:~/lstm/java/dl4j/libnd4j# export LIBND4J_HOME=`pwd`

[6-13] root@7778850ee17b:~/lstm/java/dl4j/libnd4j# cd ..

[6-14] root@7778850ee17b:~/lstm/java/dl4j# cd nd4j/

[6-15] root@7778850ee17b:~/lstm/java/dl4j/nd4j# mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:nd4j-cuda-8.0,!:nd4j-cuda-8.0-platform,!:nd4j-tests'

[6-16] root@7778850ee17b:~/lstm/java/dl4j/nd4j# cd ..

[6-17] root@7778850ee17b:~/lstm/java/dl4j# cp -rp ./nd4j/ /usr/local/lib/

[6-18] root@7778850ee17b:~/lstm/java/dl4j# cd datavec/

[6-19] root@7778850ee17b:~/lstm/java/dl4j/datavec# bash buildmultiplescalaversions.sh clean install -DskipTests -Dmaven.javadoc.skip=true

[6-20] root@7778850ee17b:~/lstm/java/dl4j/datavec# cd .. 

[6-21] root@7778850ee17b:~/lstm/java/dl4j# cd deeplearning4j/

[6-22] root@7778850ee17b:~/lstm/java/dl4j/deeplearning4j# mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:deeplearning4j-cuda-8.0'

[6-23] root@7778850ee17b:~/lstm/java/dl4j/deeplearning4j# cd ..

[6-24] root@7778850ee17b:~/lstm/java/dl4j# cd ..

[6-25] root@7778850ee17b:~/lstm/java# cd ..

[6-26] root@7778850ee17b:~/lstm# cd lstm_examples/

[6-27] root@7778850ee17b:~/lstm/lstm_examples# cd dl4j-lstm/

[6-28] root@7778850ee17b:~/lstm/lstm_examples/dl4j-lstm# cd dl4j/

[6-29] root@7778850ee17b:~/lstm/lstm_examples/dl4j-lstm/dl4j# mvn clean compile

[6-30] root@7778850ee17b:~/lstm/lstm_examples/dl4j-lstm/dl4j# mvn clean package 

[6-31] root@7778850ee17b:~/lstm/lstm_examples/dl4j-lstm/dl4j# java -Djava.library.path=/usr/lib/:/usr/lib/gcc/x86_64-linux-gnu/5/:/usr/local/lib/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native/target/classes/org/nd4j/nativeblas/linux-x86_64:/root/lstm/java/dl4j/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native/target/classes/org/nd4j/nativeblas/linux-x86_64:/root/lstm/java/dl4j/javacpp-presets/openblas/target/classes/org/bytedeco/javacpp/linux-x86_64/ -cp /root/.m2/repository/org/slf4j/slf4j-api/1.7.12/slf4j-api-1.7.12.jar:/root.m2/repository/org/slf4j/slf4j-simple/1.7.12/slf4j-simple-1.7.12.jar:/root/.m2/repository/org/nd4j/nd4j-jackson/0.7.3-SNAPSHOT/nd4j-jackson-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/nd4j/nd4j-common/0.7.3-SNAPSHOT/nd4j-common-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/nd4j/nd4j-context/0.7.3-SNAPSHOT/nd4j-context-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/nd4j/nd4j-buffer/0.7.3-SNAPSHOT/nd4j-buffer-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/nd4j/nd4j-api/0.7.3-SNAPSHOT/nd4j-api-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/deeplearning4j/deeplearning4j-core/0.7.3-SNAPSHOT/deeplearning4j-core-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/deeplearning4j/deeplearning4j-nn/0.7.3-SNAPSHOT/deeplearning4j-nn-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/deeplearning4j/deeplearning4j-nlp/0.7.3-SNAPSHOT/deeplearning4j-nlp-0.7.3-SNAPSHOT.jar::/root/.m2/repository/org/nd4j/nd4j-native-platform/0.7.3-SNAPSHOT/nd4j-native-platform-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/nd4j/nd4j-native/0.7.3-SNAPSHOT/nd4j-native-0.7.3-SNAPSHOT.jar:/root/.m2/repository/commons-lang/commons-lang/2.6/commons-lang-2.6.jar::/root/.m2/repository/commons-io/commons-io/1.3.2/commons-io-1.3.2.jar:/root/.m2/repository/org/apache/commons/commons-compress/1.8/commons-compress-1.8.jar::/root/.m2/repository/org/apache/commons/commons-math3/3.6/commons-math3-3.6.jar:/root/.m2/repository/org/nd4j/nd4j-native-api/0.7.3-SNAPSHOT/nd4j-native-api-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/nd4j/jackson/0.7.3-SNAPSHOT/jackson-0.7.3-SNAPSHOT.jar:/root/.m2/repository/com/google/collections/google-collections/1.0/google-collections-1.0.jar:/root/.m2/repository/org/reflections/reflections/0.9.10/reflections-0.9.10.jar:/root/.m2/repository/org/bytedeco/javacpp/1.3/javacpp-1.3.jar::/root/.m2/repository/commons-codec/commons-codec/1.10/commons-codec-1.10.jar:/root/.m2/repository/org/bytedeco/javacpp-presets/openblas/0.2.19-1.3/openblas-0.2.19-1.3.jar:/root/.m2/repository/org/bytedeco/javacpp-presets/1.3/javacpp-presets-1.3.jar:/root/deeplearning/nd4j/nd4j-shade/jackson/target/jackson-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/apache/commons/commons-lang3/3.5/commons-lang3-3.5.jar:/root/.m2/repository/commons-io/commons-io/2.5/commons-io-2.5.jar:/root/.m2/repository/com/google/guava/guava/21.0/guava-21.0.jar:/root/.m2/repository/org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar:/root/.m2/repository/javassist/javassist/3.12.1.GA/javassist-3.12.1.GA.jar::/root/lstm/lstm_examples/dl4j-lstm/dl4j/target/dl4j-1.0-SNAPSHOT.jar com.mycompany.lstm.App


[6-32] the output may look like 

	
	... lstm ...
	Using existing text file at /tmp/Shakespeare.txt
	Loaded and converted file: 5459809 valid characters of 5465100 total characters (5291 removed)
	SLF4J: Failed to load class "org.slf4j.impl.StaticLoggerBinder".
	SLF4J: Defaulting to no-operation (NOP) logger implementation
	SLF4J: See http://www.slf4j.org/codes.html#StaticLoggerBinder for further details.
	Number of parameters in layer 0: 223000
	Number of parameters in layer 1: 321400
	Number of parameters in layer 2: 15477
	Total number of network parameters: 559877
	 Wait for a while 
	--------------------
	Completed 10 minibatches of size 32x1000 characters
	Sampling characters from network given initialization ""
	----- Sample 0 -----
	loie Cathblk rane siol anire mawh mame; Ohufr.
	 TERLITDIP. Ior
	  G wome ming.
	   Bsmestae lurase obuves ohfollblas foon sole;
	    Thens that angiss and y underdud angin heajaonim metchin en, ilwind
	    That eomee reufane fomlond thiw ti, arow,
	    Khalg me the dend tosbitisL.COrDLTOESF
	 SLMaELOE.  Wo
	
	----- Sample 1 -----
	l., He bet Qnt in woug aduariyee anteres. bore wonoqdhas Solive mheugine.
	    Whle eeeed, thin houth yeus foulithey, me pesile wo rorher, Walile tho eeulh feolon orite he rerbud yoot.
	  uADI. Ne k's, yaud hham mageos wey mour the menre
	
	  CADHUSSOM. ANiadiln fit youthe, hit lithe woole sarot; Cowt be
	
	----- Sample 2 -----
	l!,
	  Oreod apisis rereesot mipag bilid yon . Deer,
	    Thit Sore hy moou
	    Heif mika useethr aut oree in;
	    Brale tou dis ais oure anyotaw-'fy tile theo -he hinen; Iou  hnsorheilo aad.
	   muis the were yoonked aagh ce our
	    Thace ve
	    Iraudane euryone dnet, Duugd,
	  S . ine, hu  ho  Onantor'
	
	----- Sample 3 -----
	lnte pe biiltigoan,
	             P Hoix' I Home toun im me tlole lelt WolliRt
	    he toul ousvelt amente,
	    asice youridaslet of y wime hesead thripous orekos,
	    Ant , lrat he eridhit. Car miko woud  o liod olire
	    deocd iy masmemill tomerell the lenoitery.   Aes he aweees; li! wroid hounoufe, 
	
	test_set_max is reached.
	
	Example complete
	



