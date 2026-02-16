
% set parameters
lowerBound = 1.28;
gridSpaceStart = .02;
bufferSize=10000;

%load in previously checked cases
dataStruct{1} = load('./12BinsStore_parallel1.mat');
dataStruct{2} = load('./12BinsStore_parallel2.mat');
dataStruct{3} = load('./12BinsStore_parallel3.mat');
load 12BinsStore_parallel.mat;

for i=1:3
if numBinsCompleted(dataStruct{i}.binID)<dataStruct{i}.numBinsCompleted
    numBinsCompleted(dataStruct{i}.binID)=dataStruct{i}.numBinsCompleted;
    numBinsTotal(dataStruct{i}.binID)=dataStruct{i}.numBinsTotal;
    storeWorstFinal{dataStruct{i}.binID}=dataStruct{i}.storeWorst;
end
end

save('~/12BinsStore_parallel.mat','storeWorstFinal','numBinsCompleted','numBinsTotal');

% set up parallel and gpu
c = parcluster;
c.NumWorkers=3;
p = parpool(c);

memBuffer = 600000*576;

% set max amount of energy in 1 bin, since anything larger already violates
% bound
x = sqrt(lowerBound / 3);
if x<=0.5
    error('Problem with lower bound!');
end

% read in size from previous run.  Starts at 3 bins
numMasterBins = length(storeWorstFinal);

startFromMasterBin = find(numBinsTotal-numBinsCompleted,1,'first');

% bins that will be run later
needMasterBin = find(numBinsTotal-numBinsCompleted);
needMasterBinLength=length(needMasterBin);

startFromMasterBin = startFromMasterBin+3;
endFromMasterBin = startFromMasterBin+p.NumWorkers-1;

% loop over all parent bins
for iter=(needMasterBinLength-2):p.NumWorkers:needMasterBinLength
%for iter=startFromMasterBin:p.NumWorkers:endFromMasterBin

% set paralellization
spmd
	
    % collect indices for each CPU and GPU
	gpuDevice(labindex);
%	i = iter+labindex-1;
	i = needMasterBin(iter+labindex-1);
%    fprintf(fid{labindex},'Master Bin %d of %d\n',[iter numMasterBins]);
    display(['Master Bin ' num2str(i) ' of ' num2str(numMasterBins)]);
    gridSpace = gridSpaceStart;
    weight = 1;

    

    binStore = storeWorstFinal{i};
    lengthBinStore=length(binStore);
    indicator = numBinsCompleted(i)+1;
    numBins=0;
    
    while indicator<=lengthBinStore


tic;
        % set specific parent bin to create subbins
        bin = binStore{indicator};

    % if we are just starting on a new size of parent bin, we must
    %   pre-cauclate some of the matrices
	if length(bin) ~= numBins/2

	    display('Calculating New Bin Size Matrices');
%	    fprintf(fid{labindex},'Calculating New Bin Size Matrices\n');

	    numBins = 2*length(bin);
        % create vectors of all possible combinations of f_i
        [xtmp,ytmp]=meshgrid(1:numBins,1:numBins);
        pairs = [xtmp(:) ytmp(:)];
        numPairs = size(pairs,1);
        subsetBins = gpuArray(full(sparse([(1:numPairs) (1:numPairs)],...
            [sum(pairs,2)-1;sum(pairs,2)],...
            ones(1,2*numPairs),numPairs,2*numBins)));
            
        %ship to gpu
        pairsGpu = gpuArray(pairs);
	    pairsCPU = pairs;

        % create matrix of indicator functions for which f_i*f_j contribue
        %   to a given interval in the convolution space
	    sumIndicesStore = cell(2*numBins,1);
	    binsContribute = cell(2*numBins,1);
   	    for j=2:2*numBins	
            numIntervals = 2*numBins - j + 1;
            row = [1;zeros(numIntervals-1,1,'gpuArray')];
            column = [ones(1,j,'gpuArray') zeros(1,2*numBins-j,'gpuArray')];
            convBinIntervals = toeplitz(row,column);
                   
            sumIndicesStore{j} = single((subsetBins * convBinIntervals')==2);
            binsContribute{j} = zeros(numBins,numIntervals,'single','gpuArray');
            for k=1:numIntervals
                binsContribute{j}( unique(pairs(logical(sumIndicesStore{j}(:,k)),1)) , k ) = single(1);
            end
        end
    
        % store on cpu
	    sumIndicesStoreCPU = cell(2*numBins,1);
	    for j=2:2*numBins
            sumIndicesStoreCPU{j} = gather(sumIndicesStore{j});
	    end

    end

    % store on gpu
	if ~existsOnGPU(pairsGpu)
	    pairsGpu = gpuArray(pairsCPU);
	    for j=2:2*numBins
		sumIndicesStore{j} = gpuArray(sumIndicesStoreCPU{j});
	    end
    end
	    
    % create all child bins of parent bin
    indicator = indicator+1;
    numBins=2*length(bin);

    % max size a single f_i can be before it already satisfies f_i*f_i is
    %   bigger than lowerbound
    x = sqrt(lowerBound / numBins);

    tmpPartition = cell(numBins/2,1);
    tmpLength = zeros(numBins/2,1);
    for j=1:numBins/2
        % amount of mass in parent bin f_j
        weight = bin(j);
        % high and low for splitting into two subbins
        start = round((weight-x)/gridSpace)*gridSpace;
        endPoint = round(min(weight,x)/gridSpace)*gridSpace;
        subBins = max(0,start) : gridSpace : endPoint;
        % create bin and weight-bin so that sum of two subbins = weight
        partialBin = [subBins; max(weight-subBins,0)]';
        tmpPartition{j} = single(partialBin);
        tmpLength(j) = length(subBins);
    end
        
    % store on gpu
	tmpPartition = gpuArray(cell2mat(tmpPartition));
	cumLength = cumsum(tmpLength);

    % check number of subbins to see if we need to split for memory reasons
    %   (more important for GPU than for CPU)
    numRepeats = cumprod(tmpLength);
	numRepeats(2:end+1) = numRepeats;
	numRepeats(1) = 1;
	numRepeats = single(numRepeats);

    sizeMatrix = numBins/2;
    numRows = prod(tmpLength(1:sizeMatrix));
        
	availableMem = floor(memBuffer/(numBins)^2);
	iterateRows = 1:availableMem:numRows;

	numCombos = length(iterateRows);
	iterateRows(end+1) = numRows+1;
    
	tmpBinStore = cell(1,numCombos);

    for k = 1:numCombos

        % select a subset of child bins to check now, will loop over all
	    indexMatrix = single(iterateRows(k):iterateRows(k+1)-1);
	    index = floor( (1./numRepeats(1:sizeMatrix)) * indexMatrix );

	    index = bsxfun(@mod,index,tmpLength);
	    index = bsxfun(@plus,index,[0;cumLength(1:sizeMatrix-1)]) + 1;

        % choose a collection of child bins
	    matrix_tmp = tmpPartition(index(:),:)';
	    matrix_tmp = reshape(matrix_tmp(:),[2*sizeMatrix,iterateRows(k+1)-iterateRows(k)])';


	    clear index{labindex};


        % create all possible f_i*f_j using pairs
	    functionMult = matrix_tmp(:,pairsGpu(:,1));	
        functionMult = functionMult.*matrix_tmp(:,pairsGpu(:,2));


        aboveThreshold = zeros([size(matrix_tmp,1),1],'gpuArray');
        indices = true([size(matrix_tmp,1),1],'gpuArray');
        j=2;
        stopCond = Inf;
        
        % loop over bins of size 2, 3, 4, ..., all the way to whole
        %   interval.  while loop b/c allows to stop early if all cases
        %   already greater than lower bound
        while j<=2*numBins && stopCond>0

            % multiply all f_i*f_j by which contribute to given bins
            %   only does so for subset of children that didn't get eliminted
            %   on previous size collection in convolution space
            convFunctionVals = functionMult(indices,:) * sumIndicesStore{j};
            
            % renormalize by size of group of bins in convolution space
            convFunctionVals = convFunctionVals * (2*numBins)/j;

            % set dynamic bound to beat, which is lower bound plus epsilon
            %   squared plus 2*epsilon*WeightInThatSubinterval
            boundToBeat = (lowerBound+gridSpace^2) + 2*gridSpace*( matrix_tmp(indices,:) * binsContribute{j} );
    
            % check if value is greater than bound to beat. Sum across all
            % intervals of that size (only care if this is greater than 0)
            checkBins = sum(convFunctionVals >= boundToBeat,2);

            % 1 for all children that were bigger than bound to beat
            aboveThreshold(indices) = aboveThreshold(indices) | checkBins;

            % only keep children that were less than bound to beat
            indices = aboveThreshold==0;
            stopCond = sum(indices);
            j=j+1;

        end
        
        % store all children that were less than bound.  This is collection
        %   of bins that have to be subdivided and checked at finer
        %   resolution
        leftOver = gather(matrix_tmp(aboveThreshold==0,:));
	    tmpBinStore{k} = leftOver';
	           
		clear functionMult{labindex} matrix_tmp{labindex} indices{labindex} aboveThreshold{labindex};
 

    end
	
    % collect all children that need to be subdivided across all possible
    % children of given parent bin
	tmpCell = cell2mat(tmpBinStore);
	tmpCell = num2cell(tmpCell,1);
	binStore(lengthBinStore+1 : lengthBinStore+length(tmpCell)) = tmpCell;
	lengthBinStore = lengthBinStore + length(tmpCell);
        
tStop = toc;

%	fprintf(fid{labindex},'%d %d %d %d %f\n',[i indicator-1 lengthBinStore length(bin) tStop]);
    display([num2str(i) ' '  num2str(indicator-1) ' ' num2str(lengthBinStore) ' ' num2str(length(bin)) ' ' num2str(numCombos) ' ' num2str(tStop)]);

    % sporatically save all bins that need to be checked later
    if (mod(indicator-1,100)==0) | (indicator > lengthBinStore)
        fname = ['~/12BinsStore_parallel' num2str(labindex) '.mat'];
        parConvolutionSave(fname,i,indicator-1,binStore,lengthBinStore);
        display(['Saved ' num2str(indicator-1)]);
    end
% end loop over all possible parents at given binning size
end

% end spmd loop over parallel
end

% for k=1:p.NumWorkers
%     tmpLength = lengthBinStore{k};
%     tmpStore = binStore{k};
%     storeWorstFinal{iter+k-1} = tmpStore(1:tmpLength);
%     numBinsCompleted(iter+k-1) = tmpLength;
%     numBinsTotal(iter+k-1) = tmpLength;
% end

% store info from all parents at given binning size
for k=1:p.NumWorkers
    tmpLength = lengthBinStore{k};
    tmpStore = binStore{k};
    storeWorstFinal{i{k}} = tmpStore(1:tmpLength);
    numBinsCompleted(i{k}) = tmpLength;
    numBinsTotal(i{k}) = tmpLength;
end


save('~/12BinsStore_parallel.mat','storeWorstFinal','numBinsCompleted','numBinsTotal');

% end loop over all bins that need to be refined
end

% if we reach here, we're done and bound is true!!!!
display('OMG it ended and bound is true!');