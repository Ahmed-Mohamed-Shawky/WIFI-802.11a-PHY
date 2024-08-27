classdef IEEE802_11a_Transmitter
    
    %% properties
    properties(Access = public)
        %% TXVECTOR Parameters
        LENGTH      % 1-4095
        DATARATE    % 6,9,12,18,24,36,48,54
        SERVICE = [ 1 0 1 0 1 0 1 ]; %[1;0;1;1;1;0;1];
        % TXPWR_LEVEL   % 1-8
        DebugMode   % control the transmitter output
                    % 0 -> TX_output = waveform
                    % 1 -> TX_output = waveform + signal_out + data_out
    end

    properties(Access = private)
        
    end

    %% Methods
    methods (Access = public)
        %% Class Constractore
        function obj = IEEE802_11a_Transmitter(LENGTH,DATARATE)
            switch nargin
                case 0
                    obj.LENGTH = 100;
                    obj.DATARATE = 36;
                case 1 
                    obj.LENGTH = LENGTH;
                    obj.DATARATE = 36;
                case 2
                    obj.LENGTH = LENGTH;
                    obj.DATARATE = DATARATE;
            end
        end

        %% Main Function
        function WIFI_Packet = GenerateWaveform(obj,Data)
            %% 

            %% Short Preample Waveform
            shortPreambleWaveform = Preamble2waveform(obj,'Short');
            %% Long Preample Waveform
            longPreambleWaveform = Preamble2waveform(obj,"Long");
            %% Signal Field to Waveform
            SignalOutput = signal2waveform(obj);
            %% Data to Waveform
            DataOutput = data2waveform(obj,Data);
            
            %% Packet Waveform
            if(obj.DebugMode)
            waveform = [shortPreambleWaveform;longPreambleWaveform;...
                        SignalOutput.SignalWaveform;DataOutput.DataWaveform];
            
            WIFI_Packet = struct("waveform",waveform, ...
                                 "TxLength",obj.LENGTH, ...
                                 "DebugMode",obj.DebugMode, ...
                                 "ShortPreamble",shortPreambleWaveform, ...
                                 "LongPreamble",longPreambleWaveform, ...
                                 "SignalOutput",SignalOutput, ...
                                 "DataOutput",DataOutput);
            else
                waveform = [shortPreambleWaveform;longPreambleWaveform;...
                                SignalOutput;DataOutput];
                WIFI_Packet = struct("waveform",waveform, ...
                                    "TxLength",obj.LENGTH);
            end
        end
    end

    methods(Access = private)
        %% Main Functions
        function PreambleWaveform = Preamble2waveform(obj,PreambleType)
            switch PreambleType
                case "Long"
                     %% Long Preample Waveform
                    longPreambleSequance = [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0,...
                     1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1].';
                    longPreambleFreqDomain = [zeros(6,1);longPreambleSequance;zeros(5,1)];

                    % Long preamble sequance time domain
                    longPreambleFreqDomainCShift = circshift(longPreambleFreqDomain,64/2);
                    longPreambleTimeDomain = ifft(longPreambleFreqDomainCShift);
                    %% 
                    longPreambleTimeDomain_x2 = [longPreambleTimeDomain;longPreambleTimeDomain];
                    PreambleWaveform = [longPreambleTimeDomain_x2(end-31:end);longPreambleTimeDomain_x2];
                    % PreambleWaveform = round([longPreambleTimeDomain_x2(end-31:end);longPreambleTimeDomain_x2],3);
                case "Short"
                    %% Short Preample Waveform
                    % Short preable sequance freq domain
                    shortPreambleSequance = sqrt(13/6) * [0, 0, 1+1i, 0, 0, 0, -1-1i, 0, 0, 0, 1+1i, 0, 0, 0, -1-1i, 0, 0, 0, -1-1i, 0, 0, 0, 1+1i, 0, 0, 0, 0,...
                    0, 0, 0, -1-1i, 0, 0, 0, -1-1i, 0, 0, 0, 1+1i, 0, 0, 0, 1+1i, 0, 0, 0, 1+1i, 0, 0, 0, 1+1i, 0,0].';
                    shortPreambleFreqDomain = [zeros(6,1);shortPreambleSequance;zeros(5,1)];

                    % short preamble sequance time domain
                    shortPreambleFreqDomainCShift = circshift(shortPreambleFreqDomain,64/2);
                    shortPreambleTimeDomain = ifft(shortPreambleFreqDomainCShift);
                    PreambleWaveform = [shortPreambleTimeDomain;shortPreambleTimeDomain;shortPreambleTimeDomain(1:32)];
            end
        end  
        
        function SignalOutput = signal2waveform(obj)
            %% Signal Field to Waveform
                       % Rate NBPSC NCBPS   %R1–R4
            % RateBits = [  6   , 1 1 0 1  
            %               9   , 1 1 1 1  
            %               12  , 0 1 0 1
            %               18  , 0 1 1 1
            %               24  , 1 0 0 1
            %               36  , 1 0 1 1
            %               48  , 0 0 0 1
            %               54  , 0 0 1 1 ];
            
            % RateIndex = RateBits(:,1)==DATARATE;
            % rateBits = RateBits(RateIndex,(2:end))';
            
            switch obj.DATARATE
                case 6
                    rateBits = [1;1;0;1];
                case 9
                    rateBits = [1;1;1;1];
                case 12
                    rateBits = [0;1;0;1];
                case 18
                    rateBits = [0;1;1;1];
                case 24
                    rateBits = [1;0;0;1];
                case 36
                    rateBits = [1;0;1;1];
                case 48
                    rateBits = [0;0;0;1];
                case 54
                    rateBits = [0;0;1;1];
            end

            NBPSC = 1;
            NCBPS = 48;
            
            
            
            lengthBits = fliplr((dec2bin(obj.LENGTH,12)-'0'))';
            parity = mod(sum([rateBits;lengthBits]),2);
            
            signalFieldBits = [rateBits;0;lengthBits;parity;zeros(6,1)];
            
            
            %% Signal Field Encoding
            Trellis = poly2trellis(7,[133,171]); % K = 7 shift register + 1 % gen pol [133,171]
            encodedSignal = IEEE802_11a_Transmitter.Convolutional_encoder(signalFieldBits,Trellis,1/2)';
            %% Signal Field Interleaving
            interleavedSignal = IEEE802_11a_Transmitter.Interleaver(encodedSignal,NBPSC,NCBPS);
            %% Signal Field Mapping
            mappedSignal = IEEE802_11a_Transmitter.QAM_MOD(interleavedSignal,2);
            %% Signal Field Freq Doamin
            signalFreqDoamin = IEEE802_11a_Transmitter.Guard_Pilotes_Adder(mappedSignal,1);
            %% Signal Field Time Domain
            freqDomainShefted = circshift(signalFreqDoamin,64/2);
            timeDomain = ifft(freqDomainShefted);
            timeDomainCP = round([ timeDomain(end-15:end,:) ; timeDomain ],3);
            signalWaveform = reshape(timeDomainCP,1,[]).';
            
            if (obj.DebugMode)
            SignalOutput = struct("SignalWaveform",signalWaveform, ...
                                  "DataLength",obj.LENGTH, ...
                                  "SignalFieldBits", signalFieldBits, ...
                                  "EncoddedSignal", encodedSignal, ...
                                  "InterleavedSignal", interleavedSignal, ...
                                  "MappedSiganl", mappedSignal, ...
                                  "SignalFreqDomain", signalFreqDoamin);
            else
                SignalOutput = signalWaveform;
            end
        end
        
        function DataOutput = data2waveform(obj , data )
            %% 802.11a Standart Data Rates
            TX_Rates = [
            % Data Rate  Modulation  Coding Rate NBPSC NCBPS NDBPS
                6     ,     2       ,   1/2      , 1  , 48  , 24   ;  % BPSK   
                9     ,     2       ,   3/4      , 1  , 48  , 36   ;  % BPSK   
                12    ,     4       ,   1/2      , 2  , 96  , 48   ;  % QPSK   
                18    ,     4       ,   3/4      , 2  , 96  , 72   ;  % QPSK   
                24    ,     16      ,   1/2      , 4  , 192 , 96   ;  % 16-QAM 
                36    ,     16      ,   3/4      , 4  , 192 , 144  ;  % 16-QAM
                48    ,     64      ,   2/3      , 6  , 288 , 192  ;  % 64-QAM
                54    ,     64      ,   3/4      , 6  , 288 , 216  ;  % 64-QAM
            ];
            
            %% Parameter
            dataRateIndex = find(TX_Rates(:,1)== obj.DATARATE);
            samplingFreq = 20e6;    % 20 M sample/Sec
            %Data_Length_Bits = LENGTH*8;
            Intitial_State = obj.SERVICE;
            Mapping_Order = TX_Rates(dataRateIndex,2);
            Encoder_Rate = TX_Rates(dataRateIndex,3);
            NBPSC = TX_Rates(dataRateIndex,4); % Number Of Bits per Subcarrier
            NCBPS = TX_Rates(dataRateIndex,5); % Number Of Codded Bits per OFDM Symbol
            NDBPS = TX_Rates(dataRateIndex,6); % Number Of Data Bits per OFDM Symbol
            fftSize = 64;
            activeSC = 52;
            dataSC = 48;
            
            %% Data Preberation
            
            % % pading
            Nsys = ceil((16+(obj.LENGTH*8+6))/NDBPS);
            NData = Nsys*NDBPS;
            NPad = NData - (16+(obj.LENGTH*8+6));
            
            padedData = IEEE802_11a_Transmitter.HexData2PaddedBits(data,NPad);
            %Data = load('Standard\Standard Data\Packet Data.mat').Data;
            
            
            %% scrampling
            [scrambledData , ~] = IEEE802_11a_Transmitter.scrambler(Intitial_State,padedData);
            scrambledData(16+(obj.LENGTH*8)+1:16+(obj.LENGTH*8)+6+1) = 0;% Tail Bits = zeros "Need to be automated"
            
            %% Encoding
            Trellis = poly2trellis(7,[133,171]); % K = 7 shift register + 1 % gen pol [133,171]
            encodedData = IEEE802_11a_Transmitter.Convolutional_encoder(scrambledData,Trellis,Encoder_Rate);
            
            %% Interleaving
            encodedData_Reshaped = reshape(encodedData,NCBPS,Nsys);
            Interleaved_Data = zeros(size(encodedData_Reshaped));
            for i = 1:Nsys
            Interleaved_Data(:,i) = IEEE802_11a_Transmitter.Interleaver(encodedData_Reshaped(:,i),NBPSC,NCBPS);
            end
            interleavedData = reshape(Interleaved_Data,[],1);
            
            %% Freq Domain 
            %%Mapping
            mappedData = IEEE802_11a_Transmitter.QAM_MOD(interleavedData,Mapping_Order);
            mappedData = round(reshape(mappedData,dataSC,Nsys),3);
            % % Mapping ERROR Check
            % if debudMode
            %     plot(real(mappedData),imag(mappedData),'rx');
            % end
            %%Adding Pilots & Guard pands
            % Scrampling Sequance for intital state = [ 1 1 1 1 1 1 1 ]
            [ ~ , scrambleSequance ] = IEEE802_11a_Transmitter.scrambler(ones(7,1));
            pilotsPolarity = (scrambleSequance*-2)+1;
            
            freqDomain = zeros(fftSize,Nsys);
            for i = 1:Nsys
            freqDomain(:,i) = IEEE802_11a_Transmitter.Guard_Pilotes_Adder(mappedData(:,i),pilotsPolarity(mod(i,127)+1));
            end
            
            %% Time Domain
            freqDomainShefted = circshift(freqDomain,fftSize/2);
            timeDomain = ifft(freqDomainShefted);
            timeDomainCP = round([ timeDomain(end-15:end,:) ; timeDomain ],3);
            dataWaveform = reshape(timeDomainCP,1,[]).';
            
            %% Output Struct
            if(obj.DebugMode)
            DataOutput = struct("DataWaveform",dataWaveform, ...
                                "PadedDataBits", padedData, ...
                                "ScrambledData", scrambledData, ...
                                "EncodedData", encodedData', ...
                                "InterleavedData", interleavedData, ...
                                "MappedData", mappedData, ...
                                "DataFreqDomain", freqDomain );
            else
                DataOutput = dataWaveform;
            end
            
        end
    end

    methods(Access = private,Static)
        function data = HexData2PaddedBits(hexData,NPad)
            binaryData = dec2bin(reshape(hexData.', 1, []), 8) - '0';
            binaryData = fliplr(binaryData);
            in = reshape(binaryData',[],1);
            
            data = [zeros(16,1) ; in ; zeros(6+NPad,1)];
        end
        
        %% Bit Field Functions
        function [scrambled_data, scramble_seq] = scrambler(state,data)
            %scrambler
        
            % Initialize scrambled sequence vector
            scramble_seq= zeros(127,1);
            % Scramble sequence calc
            for i = 1:127
                scramble_seq(i) = xor(state(4), state(7));
                state = circshift(state, 1);
                state(1) = scramble_seq(i);
            end
        
            if nargin == 1
                %state = [1 0 1 1 1 0 1];
                scrambled_data = 0;
            else
                scramble_pading_Bits = (ceil(length(data)/127)*127)-length(data);
                data = [data ; zeros(scramble_pading_Bits,1)];
                data=reshape(data,127,[]);
                [~,Col]=size(data);
        
                % Initialize scrambled data 
                scrambled_data=zeros(127,Col);
                % Scramble data calc
                % repmat(scramble_seq,Col);
                % scrambled_data = xor(scramble_seq,scrambled_data);
                for j = 1:Col
                    scrambled_data(:,j)=xor(data(:,j),scramble_seq);
                end
                scrambled_data=reshape(scrambled_data,[],1);
                scrambled_data = scrambled_data(1:end-scramble_pading_Bits);
            end
        end
        
        function [Final_Encoded_data] = Convolutional_encoder(data,trellis,codeRate)
            %%% assuming initial state is zero
            currentState=0;
            
            %%% initializing the output
            encoded_data=[];
            
            for i=1:length(data)
                input_bit=data(i);
                %% Update next state 
                next_state=trellis.nextStates( currentState+1     ,   input_bit+1  );
                %% update the output
                output_symbol=trellis.outputs(   currentState+1     ,   input_bit+1);
                %% decimal to binary
                output_bits = de2bi(output_symbol, 2,  'left-msb');
                %% update current state
                currentState= next_state;
                %% concatenation
                encoded_data=[encoded_data output_bits];
            end
            %%apply Puncturing
            Final_Encoded_data = IEEE802_11a_Transmitter.Puncturing(encoded_data,codeRate);
        end
        
        function [Punctured_data] = Puncturing(encoded_data,codeRate)
            if codeRate==3/4
                Punctured_data=reshape(encoded_data,18,[])';
                removeVect=[4,5,10,11,16,17];
                Punctured_data(:,removeVect)=[];
                Punctured_data=reshape(Punctured_data',1,[]);
            elseif codeRate==2/3
                Punctured_data=reshape(encoded_data,12,[])';
                removeVect=[4,8,12];
                Punctured_data(:,removeVect)=[];
                Punctured_data=reshape(Punctured_data',1,[]);
            elseif codeRate==1/2
                Punctured_data=encoded_data;
            end
        end
        
        function Interleaved_Data = Interleaver(codedData,Nbpsc,Ncbps)
        
        s = max(Nbpsc/2,1);
        
        k = (0:Ncbps-1)';
        
        i = (Ncbps/16)* mod(k,16) + floor(k/16);
        
        j = s * floor(k/s) +  mod ((k + Ncbps - floor(16 * k/Ncbps)),s);
        
        Interleaved_Data_i(i+1,1)=codedData;
        Interleaved_Data(j+1,1)=Interleaved_Data_i;
        
        
        end
        
        function OFDM_FreqDomain = Guard_Pilotes_Adder(Mapped_Data,PilotsState)
        %% Adding Pilots [1 1 1 -1] with polarity depend on the scramble
        %% Sequance At SubCarriers [-7,-21,7,21] 
        %% Adding DC zero value at SubCarrier 0
        Pilots = [1;1;1;-1]*PilotsState;
        OFDM_FreqDomain = [zeros(6,1) ;                     %% 6 Guard 0 SC
                           Mapped_Data(1:5);Pilots(1);      %% -21
                           Mapped_Data(6:18);Pilots(2);     %% -7
                           Mapped_Data(19:24);0;            %% 0 DC
                           Mapped_Data(25:30);Pilots(3);    %%  7
                           Mapped_Data(31:43);Pilots(4);    %%  21
                           Mapped_Data(44:end);zeros(5,1)]; %% 5 Guard 0 SC
        end
        
        function Modulated_data = QAM_MOD(bits,M)
        
            Nbps=log2(M);
            % reshape bits
            reshapedBits=reshape(bits,Nbps,[])';
            switch Nbps
                %% BPSK
                case 1
                Kmod=1;
                Modulated_data=reshapedBits*2-1;
                %% QPSK
                case 2
                Kmod=1/sqrt(2);
                Tx_real=reshapedBits(:,1)*2-1 ;
                Tx_imag=reshapedBits(:,2)*2-1 ;
                Modulated_data=(Tx_real+1i*Tx_imag)*Kmod;
                %% 16QAM
                case 4
                Kmod=1/sqrt(10);
                mapping = [-3; -1; 3; 1];
            
                Tx_real=mapping(binaryVectorToDecimal(reshapedBits(:,1:2))+1);
                Tx_imag=mapping(binaryVectorToDecimal(reshapedBits(:,3:4))+1 );
                Modulated_data=(Tx_real+1i*Tx_imag)*Kmod;
                %% 64QAM
                case 6
                Kmod=1/sqrt(42);
                mapping = [-7;-5;-1; -3; 7; 5;1;3];
                Tx_real=mapping(binaryVectorToDecimal(reshapedBits(:,1:3))+1);
                Tx_imag=mapping(binaryVectorToDecimal(reshapedBits(:,4:6))+1 );
                Modulated_data=(Tx_real+1i*Tx_imag)*Kmod;
            end
        end

    end

end