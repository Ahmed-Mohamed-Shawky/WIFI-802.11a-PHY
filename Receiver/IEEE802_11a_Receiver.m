classdef IEEE802_11a_Receiver

    %% properties
    properties(Access = public)
        CFO_Mode            % Turn CFO On or Off
        TimeSyncMode        % Turn Time Syncronization On Off
        EqualizerMode       % Turn Equalization On or Off
        PacketDetectionMode % Turn Packet Detection On or Off
        PlotingMode         % Turn Ploting On or Off
        Tracking            % Turn Tracking On or Off

        % EqualizerType           % Equalizer Type 'ZF' - 'MMSE' ... etc
        % PacketDetectionType     % Using CrosCorr Or AutoCorr
    end

    properties(Access = private)
        Waveform    % Received Waveform

        %%RxWaveform  % Waveform buffer

        % debuging
        DebugMode
        ShortPreambleOutput
        LongPreambleOutput
        SignalOutput
        DataOutput
        

        CoarseCFO
        FineCFO
        TotalCFO
        Equalizer_Q

        waveformBuffer  % HW Real-Time waveform Register

        LENGTH      % 1-4095
        DATARATE    % 6,9,12,18,24,36,48,54

        STSWaveform % Short Preamble
        LTSWaveform % Long Preamble
        Data

    end

    properties(Constant)
       %% Parameters
       Sampling_Freq = 20e6; % 20 Msamples/sec
      
       N_DSc = 48;          % Number of Data Subcarriers
       N_PSc = 4;           % Number of Pilots Subcarriers
       N_TSc = obj.N_DSc+obj.N_PSc; % Number of Total Used Subcarriers
       FFT_Size =64;        % Number of Total Subcarriers & FFT Size
      
       T_FFT = 3.2e-6;      % us IFFT/FFT Period
       T_Short = 8e-6;      % us Short preamble Time
       T_Long = 8e-6;       % us Long preamble Time
       T_Preamble = 16e-6;  % us Short+Long Preamble (PLCP Preamble Duration)
       T_GI = 0.8e-6;       % us Guard Interval
       T_Sym = 4e-6;        % us OFDM Symbol Duration
      
       OFDM_Samples = obj.T_Sym*obj.Sampling_Freq;      % 4 u-sec
       GI_samples = obj.T_GI*obj.Sampling_Freq;         % 0.8 u-sec
       Preamble_samples = obj.T_Preamble*obj.Sampling_Freq;     % 16 u-sec

       SERVICE = [ 1 0 1 1 1 0 1 ]; %[1;0;1;1;1;0;1];
       
       %% Const Buffers
       Pilots = [1 1 1 -1];

       STDlongPreambleSequance = [ 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,...
                                            1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1].';

    end
    
    %% Methods
    methods (Access = public)

        function obj = IEEE802_11a_Receiver(Transmitter_Output)
            if(nargin == 1)
                obj.Waveform = Transmitter_Output.waveform;
                obj.DebugMode = Transmitter_Output.DebugMode;
                obj.ShortPreambleOutput = Transmitter_Output.ShortPreamble;
                obj.LongPreambleOutput = Transmitter_Output.LongPreamble;
                obj.SignalOutput = Transmitter_Output.SignalOutput;
                obj.DataOutput = Transmitter_Output.DataOutput;
            end

            obj.CFO_Mode = 0;      % Turn CFO On or Off
            obj.TimeSyncMode = 0;  % Turn Time Syncronization On Off
            obj.EqualizerMode = 0; % Turn Equalization On or Off
            obj.PacketDetectionMode = 0; % Turn Packet Detection On or Off
            obj.PlotingMode = 0; % Turn Ploting On or Off
            obj.Tracking = 0;       % Turn Tracking On or Off

            %%obj.waveformBuffer = zeros(160,1); %% HW Buffer Size

        end

        function RX_Data = ReceiveData(obj,Transmitter_Output)

            if nargin==2
                obj.Waveform = Transmitter_Output.wavedorm;
                obj.DebugMode = Transmitter_Output.DebugMode;
                obj.ShortPreambleOutput = Transmitter_Output.ShortPreamble;
                obj.LongPreambleOutput = Transmitter_Output.LongPreamble;
                obj.SignalOutput = Transmitter_Output.SignalOutput;
                obj.DataOutput = Transmitter_Output.DataOutput;
            end
            
%% ----------------------------------------------------------------------------------- 
            %% Short Preamble
            ShortPreamble_State();
            %% Packet Detection
            if obj.PacketDetectionMode
                %% SignalDetection

                %% PacketDetection

                %% PacketSync
                                
            end
            %% Coars Estimation
            if obj.CFO_Mode
                

            end    
%% ----------------------------------------------------------------------------------- 
            %% Long Preamble
            % longPreambleWaveform = preambleWaveforms((Preamble_samples/2)+1:Preamble_samples);
            if RX_State.CFO_Mode
                % if(RX_State.ConstlationPlot)
                % %% Display Fine CFO Error
                % disp('Estimated Fine Epsilon: ');disp(Fine_Esti_Epsilon);
                % %% Display Total CFO Error
                % disp('Estimated CFO Epsilon: ');disp(Fine_Esti_Epsilon+Coarse_Esti_Epsilon);
                % end

            end
            
            % longPreambleNoCP = longPreambleWaveform(33:end);
            % longPreamble_reshaped = reshape(longPreambleNoCP,64,2);
            % longPreamble_FD = circshift(fft(longPreamble_reshaped),32);
            % longPreambleSequance = Gaurd_Remover(longPreamble_FD);
            
            if RX_State.EqualizerMode

                % dataInput.estmatedChannel = EstmatedChannel;
                % signalInput.estmatedChannel = EstmatedChannel;
            
                % if RX_State.ConstlationPlot
                %     figure();
                %     subplot(2,1,1);
                %     plot(long_1,'bx');hold on;
                %     plot(long_2,'bx');hold on;
                %     plot(STDlongPreambleSequance',zeros(52),'ro','LineWidth',2);
                %     title("Long Preambles");
                %     % plot(1:N_TSc,STDlongPreambleSequance,'o');hold on;
                %     % plot(1:N_TSc,ZF_long_1,'x');
                %     subplot(2,1,2);
                %     plot(1:N_TSc,abs(EstimatedChannel));
                %     title('Estimated Channel')
                % end
            end
            
                % longPreamble_2 = 
%% ----------------------------------------------------------------------------------- 
            %% SignalField paramters
            [obj.DATARATE , obj.LENGTH] = Signal_State(signalInput);
%% ----------------------------------------------------------------------------------- 
            %% DATA Field
            RX_Data = Data_State();
        end

    end

    methods (Access = private)
        function  ShortPreamble_State(obj)

        end

        function LongPreamble_State(obj)

        end

        %% Waveform to Data Functions
        function obj = Signal_State(obj)
            %% Signal Field paramters Extraction Function

            %% CFO Correction
            % if RX_State.CFO_Mode
            %    signalWaveform = signalWaveform .* signalInput.CoarseCFO;
            %    signalWaveform = signalWaveform .* signalInput.FineCFO;
            % end
            
            %% Signal Freq Domain
            
            %% Time Sync
            % if RX_State.TimeSyncMode
            % 
            %     %  OFDM symbol + CP 
            %     Signal_GI  = signalWaveform(1:16);
            % 
            %     % cross-correlation between CP and each OFDM symbol
            %     Correlation_result = (xcorr(signalWaveform , Signal_GI));
            %     Correlation_result = Correlation_result(length(signalWaveform)-length(Signal_GI)+1 :end) ; 
            %     norm_factor        = sqrt((sum(abs(Signal_GI).^2))* (sum(abs(signalWaveform).^2)));
            %     Correlation_result = abs(Correlation_result)/ norm_factor;
            %     Value                 = max(Correlation_result) ;
            %     threshold             = Value*RX_State.PacketDetection.TimeSync_Threshod; 
            %     Signal_Start            = find((Correlation_result )>threshold);
            % 
            %     % start % END of  each symbol 
            %     symbolStartIndices = Signal_Start(1)+1; %% Take the First Peak (Start Peak) as a triger    
            %     symbolENDtIndices  = symbolStartIndices+64 -1; 
            % 
            %     signalWaveformNoCP   = signalWaveform(symbolStartIndices:symbolENDtIndices);
            %     signalFreqDoamin = circshift(fft(signalWaveformNoCP),FFT_Size/2);
            % 
            %     if RX_State.ConstlationPlot
            %         figure ; 
            %         subplot(2,1,1)
            %         plot(linspace(0,length(Correlation_result),length(Correlation_result)),Correlation_result);
            %         title('crosscorrelation between CP and OFDM-Symbols');
            %         subplot(2,1,2)
            %         plot(linspace(0,length(signalWaveform),length(signalWaveform)),signalWaveform);
            %         title('crosscorrelation between CP and OFDM-Symbols');
            % 
            %     end
            % 
            % else
            %     signalWaveformNoCP = signalWaveform(GI_samples+1:end);
            %     signalFreqDoamin = circshift(fft(signalWaveformNoCP),FFT_Size/2);
            % end
            
            % Freq Domain Error
            if obj.DebugMode
                Signal_FreqDomain_Error = sum(round(signalFreqDoamin,1)~=round(TX_Signal_Output.SignalFreqDomain,1))
                % find(round(signalFreqDoamin,1)~=round(TX_Signal_Output.SignalFreqDomain,1))
            end
            %% Signal Gaurd Remove & Pilots Extraction
            [signalActiveSC] = Gaurd_Remover(signalFreqDoamin);
            
            %% Equalizing
            if RX_State.EqualizerMode
                signalActiveSC = signalActiveSC .* signalInput.Equalizer_Q;
            end
            
            mappedSignal = PilotsExtraction(signalActiveSC);
            % Gaurd Removing and Pilot Extraction Error
            % if RX_State.DebugMode
            %     % Signal_G
            % end
            %% Signal Demapping
            interleavedSignal = QAM_DEMOD(round(mappedSignal),2); %%*************%%
            % Demapping Error
            if RX_State.DebugMode
                Signal_Demapping_Error = sum(interleavedSignal~=TX_Signal_Output.InterleavedSignal)
                find(interleavedSignal~=TX_Signal_Output.InterleavedSignal)
            end
            %% Signal DeInterleaving
            enCoddedSignal = deInterleaver(interleavedSignal',1,48);
            % DeInterleaving Error
            if RX_State.DebugMode
                Signal_DeInterleaver_Error = sum(enCoddedSignal'~=TX_Signal_Output.EncoddedSignal)
                find(enCoddedSignal'~=TX_Signal_Output.EncoddedSignal)
            end
            %% Signal Decoding
            trellis = poly2trellis(7, [133 171]);
            signalBits = viterbi_decoder(enCoddedSignal',trellis,48,1/2);
            % Decodding Error
            if RX_State.DebugMode
                Signal_Decoding_Error = sum(signalBits~=TX_Signal_Output.SignalFieldBits)
                find(signalBits~=TX_Signal_Output.SignalFieldBits)
            end
            %% Signal pareamters
            RateBits = [  6   , 1 1 0 1  
                          9   , 1 1 1 1  
                          12  , 0 1 0 1
                          18  , 0 1 1 1
                          24  , 1 0 0 1
                          36  , 1 0 1 1
                          48  , 0 0 0 1
                          54  , 0 0 1 1 ];
            
            % Data Rate
            RATE_Bits = signalBits(1:4);
            RATE_Decimal = bin2dec(num2str(RATE_Bits'));
            RateIndex = bin2dec(num2str(RateBits(:,2:end)))==RATE_Decimal;
            DATARATE = RateBits(RateIndex); % 6,9,12,18,24,36,48,54
            % Data Lenght
            LENGTH_Bits = signalBits(17:-1:6)';
            LENGTH_Decimal = bin2dec(num2str(LENGTH_Bits ));
            LENGTH = LENGTH_Decimal;    % 1-4095
        end
        %% -----------------------------------------------------------------
        function Data = Data_State(obj)
            
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
            
            dataRateIndex = find(TX_Rates(:,1)==dataInput.DATARATE);
            %% Data Demodulation parameters
            Mapping_Order = TX_Rates(dataRateIndex,2);
            Encoder_Rate = TX_Rates(dataRateIndex,3);
            NBPSC = TX_Rates(dataRateIndex,4); % Number Of Bits per Subcarrier
            NCBPS = TX_Rates(dataRateIndex,5); % Number Of Codded Bits per OFDM Symbol
            NDBPS = TX_Rates(dataRateIndex,6); % Number Of Data Bits per OFDM Symbol
            
            Nsys = ceil((16+(8*dataInput.LENGTH)+6)/NDBPS);
            Ndata = Nsys*NDBPS;
            Npad = Ndata-(16+(8*dataInput.LENGTH)+6);
            
            
            %% Data Freq Domain
            dataWaveforms = dataInput.dataWaveform;
            
            %% Coarse Correction
            if RX_State.CFO_Mode
                
                %% Before Coarse Correction
                if RX_State.ConstlationPlot && ~RX_State.EqualizerMode && ~RX_State.PacketDetection.DetectionMode
                    CFO_dataWaveforms = reshape(dataWaveforms,OFDM_Samples,Nsys);
                    CFO_dataWaveformNoCP = CFO_dataWaveforms(GI_samples+1:end,:);
                    CFO_dataFreqDoamin = circshift(fft(CFO_dataWaveformNoCP),FFT_Size/2);
                    CFO_dataActiveSC = Gaurd_Remover(CFO_dataFreqDoamin);
            
                    figure;
                    plot(CFO_dataActiveSC, 'ro', 'LineWidth', 1);
                    % plot(real(y_coarse),imag(y_coarse), 'ro', 'LineWidth', 1);
                    title('Constellation Diagram for RX-Waveform Before Coarse CFO correction');
                end
                
                %% Coarse Correction
                dataWaveforms = dataWaveforms .* dataInput.CoarseCFO;
            
                %% After Coarse Correction
                if RX_State.ConstlationPlot && ~RX_State.EqualizerMode && ~RX_State.PacketDetection.DetectionMode
                    CFO_dataWaveforms = reshape(dataWaveforms,OFDM_Samples,Nsys);
                    CFO_dataWaveformNoCP = CFO_dataWaveforms(GI_samples+1:end,:);
                    CFO_dataFreqDoamin = circshift(fft(CFO_dataWaveformNoCP),FFT_Size/2);
                    CFO_dataActiveSC = Gaurd_Remover(CFO_dataFreqDoamin);
            
                    figure;
                    plot(CFO_dataActiveSC, 'rx', 'LineWidth', 1);
                    % plot(real(y_coarse),imag(y_coarse), 'ro', 'LineWidth', 1);
                    title('Constellation Diagram for RX-Waveform after Coarse CFO correction');
                end
                
                %% Fine Correction
                dataWaveforms = dataWaveforms .* dataInput.FineCFO;
            
                %% After Coarse Correction
                if RX_State.ConstlationPlot %&& ~RX_State.EqualizerMode && ~RX_State.PacketDetection.DetectionMode
                    CFO_dataWaveforms = reshape(dataWaveforms,OFDM_Samples,Nsys);
                    CFO_dataWaveformNoCP = CFO_dataWaveforms(GI_samples+1:end,:);
                    CFO_dataFreqDoamin = circshift(fft(CFO_dataWaveformNoCP),FFT_Size/2);
                    CFO_dataActiveSC = Gaurd_Remover(CFO_dataFreqDoamin);
            
                    figure;
                    plot(CFO_dataActiveSC, 'rx', 'LineWidth', 1);
                    % plot(real(y_coarse),imag(y_coarse), 'ro', 'LineWidth', 1);
                    title('Constellation Diagram for RX-Waveform after Fine CFO correction');
                end
            
            end
            
            %% Tracking
            if RX_State.Tracking
                %% Tracking With CP
                if ~RX_State.Tracking_Mode
                    OFDM_Symbols = zeros(OFDM_Samples,Nsys);
                    omega_est = 0;
                    estimated_freq_offset = 0;
                    for i = 1:Nsys
                        symbolStart  = 1 + (i - 1) * OFDM_Samples ;
                        symbolEnd    = i * OFDM_Samples  ; 
                    
                        if symbolEnd > length(dataWaveforms)
                            break;  
                        end
                        ONE_RX_OFDM_Symbol = dataWaveforms(symbolStart:symbolEnd);
                    
                        % Feedback
                        alpha = 0.08; 
                        N = 64 ; 
                        freq_est = 0 ; 
                        for k=1:16
                            freq_est = freq_est + ONE_RX_OFDM_Symbol(k+N).*conj(ONE_RX_OFDM_Symbol(k)) ;
                        end 
                        omega_est = alpha *omega_est  + ((angle((freq_est)))/((2*pi*80)));
                        fprintf('Estmated Omega for OFDM Num: %d',i);
                        disp(omega_est);
                        disp(omega_est+dataInput.Estemated_Omega);
            
                        %% Phase Tracking correction 
                        n = (0:length(ONE_RX_OFDM_Symbol)-1 ) ;
                        Phase = exp(-1j* 2 * pi * n *omega_est) ; 
                        ONE_RX_OFDM_Symbol = ONE_RX_OFDM_Symbol.*Phase.' ; 
                        % OFDM_Symbols(:,i)   = ONE_RX_OFDM_Symbol;  
            
                        %% Pilot-Aided Frequency Offset Estimation
            
                        Tracking_dataWaveformNoCP = ONE_RX_OFDM_Symbol(GI_samples+1:end,:);
                        Tracking_dataFreqDoamin = circshift(fft(Tracking_dataWaveformNoCP),FFT_Size/2);
                        Tracking_dataActiveSC = Gaurd_Remover(Tracking_dataFreqDoamin);
                        [~ , dataPilots] = PilotsExtraction(Tracking_dataActiveSC);
            
                        [ ~ , scrambleSequance ] = scrambler(ones(7,1));
                        pilotsPolarity = (scrambleSequance*-2)+1;
                        STD_DataPilots = kron(Pilots,pilotsPolarity');
            
                        alpha= 0.02;
            
                        epsilon_hat = 0;
                        for k = 1:4
                            epsilon_hat = epsilon_hat + (Pilots*STD_DataPilots(i+1)) * (conj(dataPilots));
                        end
                
                        % estimate frequency offset
                        estimated_freq_offset = alpha*estimated_freq_offset +(1/(2*pi*64))*angle(epsilon_hat);
                
                        % time domain correction
                        n = (0:OFDM_Samples-1); 
                        Phase = exp(-1j * 2 * pi * n * estimated_freq_offset); 
                        correctedSignal = ONE_RX_OFDM_Symbol .* Phase.'; 
                        OFDM_Symbols(:,i) =  correctedSignal;
                    end
                    if RX_State.ConstlationPlot
                    % Tracking_dataWaveforms = reshape(dataWaveforms,OFDM_Samples,Nsys);
                    Tracking_dataWaveformNoCP = OFDM_Symbols(GI_samples+1:end,:);
                    Tracking_dataFreqDoamin = circshift(fft(Tracking_dataWaveformNoCP),FFT_Size/2);
                    Tracking_dataActiveSC = Gaurd_Remover(Tracking_dataFreqDoamin);
            
                    figure;
                    plot(Tracking_dataActiveSC, 'x', 'LineWidth', 1);
                    % plot(real(y_coarse),imag(y_coarse), 'ro', 'LineWidth', 1);
                    title('Constellation Diagram for RX-Waveform after CP Tracking correction');
                    end
                end
            end
            
            dataWaveforms = reshape(dataWaveforms,OFDM_Samples,Nsys);
            dataWaveformNoCP = dataWaveforms(GI_samples+1:end,:);
            dataFreqDoamin = circshift(fft(dataWaveformNoCP),FFT_Size/2);
            
            % Freq Domain Error
            if RX_State.DebugMode
                Data_FreqDomain_Error = sum(sum(round(dataFreqDoamin,1)~=round(TX_Data_Output.DataFreqDomain,1)))
                % find(round(signalFreqDoamin,1)~=round(TX_Signal_Output.SignalFreqDomain,1))
            end
            %% Data Gaurd Remove & Pilots Extraction
            
            dataActiveSC = Gaurd_Remover(dataFreqDoamin);
            
            %% Equalization
            if RX_State.EqualizerMode
                %% Plot Data Simpols befor equlization
                if RX_State.ConstlationPlot
                    figure()
                    plot(dataActiveSC,'x')
                    title('Before Equalizing')
                end
                
                %% Equalizing
                dataActiveSC = dataActiveSC .* dataInput.Equalizer_Q;
            
                %% Plot Data Simpols after equlization
                if RX_State.ConstlationPlot
                    figure()
                    plot(dataActiveSC,'x')
                    % title(RX_State.Equalizer)
                    title('After ZF Equalizing')
                end
            end

            [mappedData , dataPilots] = PilotsExtraction(dataActiveSC);
            
            % Scrampling Sequance for intital state = [ 1 1 1 1 1 1 1 ]
            
            % Data Subcarriers Error
            % if debugMode
            %     plot(real(round(mappedData,4)), ...
            %         imag(round(mappedData,4)),'bx');hold on;
            %     plot(real(TX_Data_Output.mappedData), ...
            %         imag(TX_Data_Output.mappedData),'ro');
            %     % Data_Subcarriers_Error = sum(round(mappedData) ~= TX_Data_Output.mappedData)
            %     Data_Pilots_Subcarriers_Error = sum(round(dataPilots)~=STD_DataPilots(:,2:Nsys+1))'
            % end
            %% Data DeMapping
            mappedData = reshape(mappedData,N_DSc*Nsys,1);
            interleavedData = QAM_DEMOD(mappedData,Mapping_Order)'; %%*************%%
            % Data Demapping Error
            if RX_State.DebugMode
                Data_Demapping_Error = sum(interleavedData'~=TX_Data_Output.InterleavedData)
            end
            
            %% Data DeInterleaving
            InterleavedData_Reshaped = reshape(interleavedData,NCBPS,Nsys);
            enCoddedData = zeros(size(InterleavedData_Reshaped));
            for OFDM_Index = 1:Nsys
                enCoddedData(:,OFDM_Index) = deInterleaver(InterleavedData_Reshaped(:,OFDM_Index),NBPSC,NCBPS);
            end
            enCoddedData = reshape(enCoddedData,NCBPS*Nsys,1);
            % Data DeInterleaving Error
            if RX_State.DebugMode
                Data_DeInterleaving_Error = sum(enCoddedData'~=TX_Data_Output.EncodedData)
            end
            
            %% Data DeCoding
            trellis = poly2trellis(7, [133 171]);
            scrambledData = viterbi_decoder(enCoddedData,trellis,NCBPS*Nsys,Encoder_Rate);
            % Data Decoding Error
            if RX_State.DebugMode
                Data_Decoding_Error = sum(scrambledData~=TX_Data_Output.ScrambledData)
            end
            %% Data DeScrampling
            initial_state=[ 1 0 1 1 1 0 1 ]'; % Need to be estimated
            padedData = scrambler(initial_state, scrambledData);
            % Data DeScrampling Error
            if RX_State.DebugMode
                Data_DeScrampling_Error = sum(padedData~=TX_Data_Output.PadedDataBits)
                % find(padedData~=TX_Data_Output.PadedDataBits)
            end
            %% Data Output
            dataBits = padedData(17:end-(Npad+6)); % Remove Service & Paded Bits
            % RX_Output = dataBits;
            RX_Data = flip(reshape(dataBits,8,dataInput.LENGTH));
            % RX_Output = RX_Data';
            Data = bin2dec(num2str(RX_Data'));
            % RX_Data_Uint8 = reshape(RX_Data_Uint8,5,[])';
        end
    end

    methods(Access = private,Static,Hidden)
        
        %% System Block Functions
        function SignalDetection()
            
        % simple threshold
        end
        %% -----------------------------------------------------------------
        function PacketDetection()

        end
        %% -----------------------------------------------------------------
        function PacketSync()

        end
        %% -----------------------------------------------------------------
        function CoarseCFOestmation()
            % Coarse CFO Estimation
            var1 = sum(imag(shortPreamble_FD(:,2) .* conj(shortPreamble_FD(:,1))));
            var2 = sum(real(shortPreamble_FD(:,2) .* conj(shortPreamble_FD(:,1))));
            Coarse_Esti_Epsilon = atan2(var1, var2) / (2 * pi*16);
            if(RX_State.ConstlationPlot)
            disp('Estimated Coarse Epsilon: ');disp(Coarse_Esti_Epsilon);
            end
        end
        %% -----------------------------------------------------------------
        function FineCFOestmation()
            % Fine CFO Estimation
            CFO_longPreambleNoCP = longPreambleWaveform(33:end);
            CFO_longPreamble_reshaped = reshape(CFO_longPreambleNoCP,64,2);
            CFO_longPreamble_FD = fft(CFO_longPreamble_reshaped);
           
            var1 = sum(imag(CFO_longPreamble_FD(:,2) .* conj(CFO_longPreamble_FD(:,1))));
            var2 = sum(real(CFO_longPreamble_FD(:,2) .* conj(CFO_longPreamble_FD(:,1))));
            Fine_Esti_Epsilon = atan2(var1, var2) / (2 * pi*64);
        end
        %% -----------------------------------------------------------------
        function CFO_Correction()
            %% Coarse CFO Correction
            k = 0:length(waveform)-1;
            Phase_coarse = exp(-1j * k * 2 * pi * Coarse_Esti_Epsilon).';

            %% Coarse Correction
            longPreambleWaveform = longPreambleWaveform .* Long_Phase_coarse;

            %% Fine CFO Correction
                k = 160:length(waveform)-1;
                Phase_Fine = exp(-1j * k * 2 * pi * Fine_Esti_Epsilon).';
                
                Long_Phase_Fine = Phase_Fine(1:Preamble_samples-160);
                Signal_Phase_Fine = Phase_Fine(Preamble_samples-160+1:Preamble_samples-160+OFDM_Samples);
                Data_Phase_Fine = Phase_Fine(Preamble_samples-160+OFDM_Samples+1:end);
            
                longPreambleWaveform = longPreambleWaveform.* Long_Phase_Fine;
                signalInput.PhaseFine = Signal_Phase_Fine;
                dataInput.PhaseFine = Data_Phase_Fine;
        end
        %% -----------------------------------------------------------------
        function ChannelEsmtation()
            % STDlongPreambleSequance = Gaurd_Remover(STDlongPreambleSequance); 
            estmatedChannel_1 = longPreambleSequance(:,1) ./ STDlongPreambleSequance;
            estmatedChannel_2 = longPreambleSequance(:,2) ./ STDlongPreambleSequance;
           
            EstimatedChannel = (estmatedChannel_2+estmatedChannel_1)/2;
           
            % else
            %     longPreambles_FD = fft(longPreamble_reshaped);
            %     longPreambleSequance = Gaurd_Remover(longPreambles_FD);
            % end
        end
        %% -----------------------------------------------------------------
        function Equalizer()
            %%Equalizer
                if RX_State.Equalizer == "ZF"
                    ZF = 1./EstimatedChannel;
                    long_1 = (longPreambleSequance(:,1).*ZF);
                    long_2 = (longPreambleSequance(:,2).*ZF);
            
                    signalInput.Equalizer_Q = ZF;
                    dataInput.Equalizer_Q = ZF;
            
                elseif RX_State.Equalizer == "MMSE"
                    % MMSE Equalizer
                    % Wiener filter for reducing noise (based on MMSE in freq.-domain)
                    % MMSE = conj(EstmatedChannel) ./ (abs(EstmatedChannel).^2 + abs(NoisePower / WaveformPower));
                    % MMSE = pinv((abs(EstmatedChannel).^2 + (NoisePower / WaveformPower))) .* EstmatedChannel';
                    % MMSE = conj(EstmatedChannel) ./ (abs(EstmatedChannel).^2 + NoisePower / WaveformPower);
                    long_avg = (longPreambleSequance(:,1)+longPreambleSequance(:,2))/2;
                    % SignalPower = (rms(longPreambleSequance(:,1))+rms(longPreambleSequance(:,2)))/2;
                    SignalPower = rms(waveform);
                    SNR_db = log10(RX_State.SNR_linear);
                    NoisePower = rms(waveform)/(10^(SNR_db/20));
                    % MMSE = conj(EstmatedChannel) ./ (vecnorm(EstmatedChannel,2,2).^2+(NoisePower / SignalPower));
                    MMSE = conj(EstimatedChannel)*SignalPower ./ ( (vecnorm(EstimatedChannel,2,2).^2)*SignalPower + NoisePower);
                    % Apply MMSE equalizer to the Rx Long Preamble waveform
                    long_1 = longPreambleSequance(:,1) .* MMSE;
                    long_2 = longPreambleSequance(:,2) .* MMSE';
            
                    signalInput.Equalizer_Q = MMSE;
                    dataInput.Equalizer_Q = MMSE;
                end
        end
        %% -----------------------------------------------------------------
        function TrackingPhaseEstmation()

        end
        %% -----------------------------------------------------------------
        function TrackingPhaseCorrection

        end
        
        %% Preambels Generation Functions
        function shortPreambleWaveform = shortPreamble2waveform()
            %% Short Preample Waveform
            % Short preable sequance freq domain
            shortPreambleSequance = sqrt(13/6) * [0, 0, 1+1i, 0, 0, 0, -1-1i, 0, 0, 0, 1+1i, 0, 0, 0, -1-1i, 0, 0, 0, -1-1i, 0, 0, 0, 1+1i, 0, 0, 0, 0,...
            0, 0, 0, -1-1i, 0, 0, 0, -1-1i, 0, 0, 0, 1+1i, 0, 0, 0, 1+1i, 0, 0, 0, 1+1i, 0, 0, 0, 1+1i, 0,0].';
            shortPreambleFreqDomain = [zeros(6,1);shortPreambleSequance;zeros(5,1)];
            
            % short preamble sequance time domain
            shortPreambleFreqDomainCShift = circshift(shortPreambleFreqDomain,64/2);
            shortPreambleTimeDomain = ifft(shortPreambleFreqDomainCShift);
            shortPreambleWaveform = [shortPreambleTimeDomain;shortPreambleTimeDomain;shortPreambleTimeDomain(1:32)];
        end
        %% -----------------------------------------------------------------
        function longPreambleWaveform = longPreamble2waveform()
            %% Long Preample Waveform
            longPreambleSequance = [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0,...
             1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1].';
            longPreambleFreqDomain = [zeros(6,1);longPreambleSequance;zeros(5,1)];
            
            % Long preamble sequance time domain
            longPreambleFreqDomainCShift = circshift(longPreambleFreqDomain,64/2);
            longPreambleTimeDomain = ifft(longPreambleFreqDomainCShift);
            %% 
            longPreambleTimeDomain_x2 = [longPreambleTimeDomain;longPreambleTimeDomain];
            longPreambleWaveform = round([longPreambleTimeDomain_x2(end-31:end);longPreambleTimeDomain_x2],3);
        end

        %% Bit Field Functions
        function [Demodulated_data] = QAM_DEMOD(Received_Data,M)
            Nbps=log2(M);   %% number of bits per symbol

            %% function parameters
            % Kmod , Threshold & output
            switch Nbps
                case 1
                    Kmod=1;
                    % divide by Kmod
                    Received_Data=Received_Data/Kmod;
                    % set inphase component
                    I=real(Received_Data);
                    % apply threshold
                    I(I>0)=1;
                    I(I<0)=0;
                    % assign to Rx
                    Demodulated_data=I;
                case 2
                    Kmod=1/sqrt(2);
                    % divide by Kmod
                    Received_Data=Received_Data/Kmod;
                    % set inphase and quadrature components
                    I=real(Received_Data);
                    Q=imag(Received_Data);
                    % apply threshold for both components
                    I(I>0)=1;
                    I(I<0)=0;
                    
                    Q(Q>0)=1;
                    Q(Q<0)=0;
                    % assign to Rx
                    Rx=[I Q];
                    % demodulated data
                    Demodulated_data=reshape(Rx',length(Received_Data)*Nbps,1);
                    
                case 4
                    Kmod=1/sqrt(10);
                    % divide by Kmod
                    Received_Data=Received_Data/Kmod;
                    % set inphase and quadrature components
                    I=real(Received_Data);
                    Q=imag(Received_Data);
                    % apply threshold for both components
                    I(I>2        )=2;
                    I(I<2 & I>0 )=3;
                    I(I<0 & I>-2)=1;
                    I(I<-2       )=0;
                    
                    Q(Q>2        )=2;
                    Q(Q<2 & Q>0 )=3;
                    Q(Q<0 & Q>-2)=1;
                    Q(Q<-2       )=0;
                    % assign to Rx
                    Rx=[decimalToBinaryVector(I,Nbps/2),decimalToBinaryVector(Q,Nbps/2)];
                    % demodulated data
                    Demodulated_data=reshape(Rx',length(Received_Data)*Nbps,1);
                case 6
                    Kmod=1/sqrt(42);
                    % divide by Kmod
                    Received_Data=Received_Data/Kmod;
                    % set inphase and quadrature components
                    I=real(Received_Data);
                    Q=imag(Received_Data);
                    % apply threshold for both components
                    I(I>6          )=4;
                    I(I<6  & I>4  )=5;
                    I(I<4  & I>2  )=7;
                    I(I<2  & I>0  )=6;
                    I(I<0  & I>-2 )=2;
                    I(I<-2 & I>-4 )=3;
                    I(I<-4 & I>-6 )=1;
                    I(I<-6         )=0;
                    
                    Q(Q>6          )=4;
                    Q(Q<6  & Q>4  )=5;
                    Q(Q<4  & Q>2  )=7;
                    Q(Q<2  & Q>0  )=6;
                    Q(Q<0  & Q>-2 )=2;
                    Q(Q<-2 & Q>-4 )=3;
                    Q(Q<-4 & Q>-6 )=1;
                    Q(Q<-6         )=0;
                    % assign to Rx
                    Rx=[decimalToBinaryVector(I,Nbps/2),decimalToBinaryVector(Q,Nbps/2)];
                    % demodulated data
                    Demodulated_data=reshape(Rx',length(Received_Data)*Nbps,1);
            end
        end
        %% -----------------------------------------------------------------
        function activeSubCarriers = Gaurd_Remover(OFDM_Symbol_FreqDomain)
            % GAURD_REMOVER
            activeSubCarriers = OFDM_Symbol_FreqDomain(7:59,:);
            activeSubCarriers = [activeSubCarriers(1:26,:);  % 23
                                activeSubCarriers(28:end,:)]; 
        end
        %% -----------------------------------------------------------------
        function [Data_SC , Pilots] = PilotsExtraction(activeSubCarriers)
            % Pilot Index ( 12 - 26 - 40 - 54 ) DC index = 33
            Data_SC = [ activeSubCarriers(1:5,:);   % 6
                        activeSubCarriers(7:19,:);  % 20
                        % activeSubCarriers(22:27,:);  % 21
                        activeSubCarriers(21:32,:);  % 33
                        activeSubCarriers(34:46,:);% 47
                        activeSubCarriers(48:end,:)];% 47
        
            Pilots = [activeSubCarriers([6,20,33,48],:)];
        end
        %% -----------------------------------------------------------------
        function deinterleavedData = deInterleaver(codedData,Nbpsc,Ncbps)
            s = max(Nbpsc/2,1);
            j=0:Ncbps-1;
            i = s * floor(j/s) +  mod((j + floor(16 * j/Ncbps))  ,s);
            k = 16 * i - (Ncbps - 1)*floor(16 * i/Ncbps);
            deinterleavedData(k+1)=codedData(j+1);
        end
        %% -----------------------------------------------------------------
        function [scrambled_data, scramble_seq] = scrambler(state,data)
            %% scrambler
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

        %% Decoder Functions
        %% -----------------------------------------------------------------
        function Final_Decoded_Bits = viterbi_decoder(Demapped_data, trellis,Depth,codeRate)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Demapped Data must be a col vector
            % Final_Decoded_Bits is also a col vector
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %==============================================================%
            depuncturedData = depuncturing(Demapped_data,codeRate);
            %==============================================================%
            N_states=trellis.numStates;
            Depth=Depth*2;
            currentState=1;
            %%%depuncturing parameters
            %==============================================================%
            if codeRate==1/2
                tr_start=[2,2,2];
                tr_end=[2,2,2];
            
                ostart=[1,1,1];
                oend =[2,2,2];
            elseif codeRate==3/4
                tr_start=[2,3,1];
                tr_end=[1,3,2];
            
                ostart=[1,1,2];
                oend =[2,1,2];
            elseif codeRate==2/3
                tr_start=[2,2];
                tr_end =[1,3];
            
                ostart=[1,1];
                oend =[2,1];
            end
            %==============================================================%
            %input data processing
            N=length(depuncturedData);
            ext=(N-floor(N/Depth)*Depth);
            Npadded_zeros(ext~=0)=Depth-ext;
            Padded_data=[depuncturedData; zeros( Npadded_zeros,1)];
            reshapedData=reshape(Padded_data, Depth, [])';
            [R,~]=size(reshapedData);
            Final_Decoded_Bits=zeros(R, Depth/2);
            V=1; 
            START=1;
            END=2;
            for K=1:R
                
                %% initialization
                currentStates=currentState;
                pathMetricOfCurrentStates=0;
                %%%FinalPathMetrics is for tracing purpose
                TrellisMatrix=inf(N_states,Depth/2);
                TrellisMatrix(currentStates,1)=pathMetricOfCurrentStates;
                U=2;
                CurrentBlock=reshapedData(K,:);
            
            
                for t  =1:2:Depth
            %======================Depuncturing parameters ========================%
               adjusted_index=mod(V-4,3)+1;
                    if codeRate ==2/3
                        adjusted_index=mod(V-3,2)+1;
                    end
                data_start=ostart(adjusted_index);
                data_end =oend(adjusted_index);
                % take two bits by two bits 
                CurrentBits=Padded_data(START:END)';
            
                
                % brach metric unit
                [BranchMetric,nextStates_nonFiltered]= Branch_Metric_unit(currentStates,CurrentBits,trellis,data_start,data_end);
                
                START=START+tr_start(adjusted_index);
                END=END+tr_end(adjusted_index);
                
                % Path Metric unit
                [pathMetricAndNextStates] = Path_Metric_unit(nextStates_nonFiltered,BranchMetric,pathMetricOfCurrentStates);
                % update current states and pathmetric
                currentStates=pathMetricAndNextStates(:,1);
                pathMetricOfCurrentStates=pathMetricAndNextStates(:,2);
                % assign the pathMetric to TrellisMatrix
                TrellisMatrix(currentStates,U)=pathMetricOfCurrentStates;
                U=U+1;
                V=V+1;
                end
            % Traceback unit
            [decodedBlock,initialStateOfNextBlock] = Trace_back_unit(TrellisMatrix,trellis,U);
            % update currentState
            currentState=initialStateOfNextBlock;
            % Final decoded bits
            Final_Decoded_Bits(K,:)=decodedBlock;
            end
            % reshape Final decoded bits into one coulmn
            Final_Decoded_Bits=reshape(Final_Decoded_Bits',[],1);
            Final_Decoded_Bits( end :-1: end-Npadded_zeros/2+1)=[];
        end
        %% -----------------------------------------------------------------
        function depuncturedData = depuncturing(encodedData,codeRate)
            if codeRate==3/4
                ReshapedData=reshape(encodedData,12,[])';
                dummy_bits=zeros(length(encodedData)/12,2);
                %%%dummy bits insertion
                depuncturedData=reshape([ReshapedData(:,1:3),dummy_bits,ReshapedData(:,4:7),dummy_bits,ReshapedData(:,8:11),dummy_bits,ReshapedData(:,12)]',[],1);
            elseif codeRate==2/3
                ReshapedData=reshape(encodedData,9,[])';
                dummy_bits=zeros(length(encodedData)/9,1);
                %%%dummy bits insertion
                depuncturedData=reshape([ReshapedData(:,1:3),dummy_bits,ReshapedData(:,4:6),dummy_bits,ReshapedData(:,7:9),dummy_bits]',[],1);
            elseif codeRate==1/2
                depuncturedData=encodedData;
            end
        end
        %% -----------------------------------------------------------------
        function [BranchMetric,nextStates] = Branch_Metric_unit(currentStates,CurrentBits,trellis,data_start,data_end)
            % calculate next states & outputs
            nextStates=reshape((trellis.nextStates( currentStates,:)' +1),[],1);
            outputOfcurrentStates=reshape(trellis.outputs(currentStates,:)' ,[],1);
            outputOfcurrentStates_Binray=dec2bin(outputOfcurrentStates)-'0';
            % calculate branch metric
            BranchMetric=sum(outputOfcurrentStates_Binray(:,data_start:data_end)~= CurrentBits, 2 );
        end
        %% -----------------------------------------------------------------
        function [compareMatrix] = Path_Metric_unit(nextStates_nonFiltered,BranchMetric,pathMetricOfCurrentStates)
            % calculate path metric
            pathMetricOfNextStates=BranchMetric+reshape(repmat(pathMetricOfCurrentStates',2,1),[],1);
            
            compareMatrix=[nextStates_nonFiltered pathMetricOfNextStates];
            % select minimum pathmetric
            if length(BranchMetric)>64
            compareMatrix=sortrows(compareMatrix);
            minPathMetrics=min(reshape(compareMatrix(:,2 ),2,[]))';
            compareMatrix = [(1:64)' ,  minPathMetrics] ;
            end
        end
        %% -----------------------------------------------------------------
        function [decodedBlock,initialStateOfNextBlock] = Trace_back_unit(TrellisMatrix,trellis,U)
            % find the state with minimum path metric in the last coulmn of Trellis matrix
            [ ~,W]=min(TrellisMatrix( : ,end ));
            initialStateOfNextBlock=W;
            % initialize decodedBlock
            decodedBlock=zeros(1,U-2);
            
            for H=U-2 : -1 : 1
            %find previous state
            [previousStates ,inputBit]=find(trellis.nextStates==W-1);
            decodedBlock(H)=inputBit(1)-1;
                     if TrellisMatrix( previousStates(1),H) <= TrellisMatrix( previousStates(2),H)
                        W=previousStates(1);
                    else
                        W=previousStates(2);
                    end
            end
        end
        %% -----------------------------------------------------------------

    end

end

 %% Packet Detection old old
            % if RX_State.PacketDetection.DetectionMode
            %     %% Algorithm info -> Preamble based -> Cross Correlation -> Between STS and RX_Waveform
            %     STD_ShortPreamble = shortPreamble2waveform();
            %     STS_Waveform_N       =  STD_ShortPreamble(1:16*RX_State.PacketDetection.NumberOfShortPreambles);
            %     cross_corr_ouput     = (xcorr((waveform),STS_Waveform_N)) ;
            %     cross_corr_ouput     =  cross_corr_ouput(length(waveform) - 16+1:end);
            % 
            %     %% Maximum Normalized Correlation (MNC)
            %     norm_factor = sum(STS_Waveform_N .* conj(STS_Waveform_N)); %The power  of STS
            %     % Normalize the cross-correlation output
            %     cross_corr_ouput = abs(cross_corr_ouput).^2 / (norm_factor).^2;
            % 
            %     %% Convolution Plotting
            %     if RX_State.ConstlationPlot
            %         figure;
            %         plot(linspace(0, length(cross_corr_ouput), length(cross_corr_ouput)), abs(cross_corr_ouput));
            %         title("Cross-correlation between STS and Wi-Fi Received Packet");
            %     end
            % 
            %     %% STS index calculation
            %     STS_detection_threshold = max(abs(cross_corr_ouput)) * RX_State.PacketDetection.Detection_Threshod; 
            %     STS_xcorr_peaks = find(cross_corr_ouput >= STS_detection_threshold);
            %     disp("Peaks Index: ");disp(STS_xcorr_peaks);
            % 
            %     %% Calculate RMS of short Preamble 
            %     num_samples =160 ;%floor(length(RX_Waveform_2(1:160)) /STS_X1_samples);
            %     STS_RMS = zeros(num_samples, 1);
            %     for k = 1:num_samples
            %     sample = waveform((k-1) + 1:k);
            %     STS_RMS(k) = sqrt(mean(abs(sample).^2));
            %     end
            % 
            %     %% Calculate RMS of short Preamble 
            %     num_samples =160 ;%floor(length(RX_Waveform_2(1:160)) /STS_X1_samples);
            %     STS_RMS = zeros(num_samples, 1);
            %     for k = 1:num_samples
            %     sample = waveform((k-1) + 1:k);
            %     STS_RMS(k) = sqrt(mean(abs(sample).^2));
            %     end
            % 
            %     %% Extract peaks of cross-correlation
            %     Peaks = findpeaks(cross_corr_ouput) ;
            %     Peaks = Peaks(1:length(STS_RMS)) ;
            % 
            %     %% Short Preambles (RMS) Vs Corrolation Peaks Plotting
            %     if RX_State.ConstlationPlot
            %         figure;
            %         scatter(STS_RMS, Peaks);
            %         title('Scatter Plot of Short Preamble RMS vs Peaks of Cross-Correlation');
            %         xlabel('RMS of Short Preamble');
            %         ylabel('Peaks of Cross-Correlation');
            %     end
            % 
            %     %%  Packet Detection and SYN using STS
            %     STS_index = STS_xcorr_peaks(1) ; 
            %     fprintf("Index of starting sample after N STS Symbols:  %d\n", STS_index + 1);
            % 
            %     STS_END = STS_index + (10-RX_State.PacketDetection.NumberOfShortPreambles)*16;
            %     fprintf("Start of Long Preamble = %d\n", STS_END + 1);
            % 
            %     LTS_END_With_GI = STS_END + 160;
            %     fprintf("Start of SIGNAL FIELD = %d\n", LTS_END_With_GI + 1);
            % 
            %     SIGNAL_FIELD_Start = LTS_END_With_GI + 1;
            %     SIGNAL_FIELD_END = SIGNAL_FIELD_Start + 80 - 1;
            %     fprintf("Start of DATA Field = %d\n", SIGNAL_FIELD_END + 1);
            % 
            %     START_Packet = STS_END - 160 + 1
            %     waveform = waveform(START_Packet:end);
            % 
            % end

%% Old Packet Detection
% Power_Window = 1;
%                 Corr_Window = 46;
% 
%                 Power_out = zeros(1,length(waveform));
%                 for sample = 1:length(waveform)-Power_Window
%                    Power_out(sample) = (sum(waveform(sample:sample+Power_Window-1).*conj(waveform(sample:sample+Power_Window-1))))/Power_Window;
%                 end
%                 Power_out_Norm = Power_out/max(Power_out);
%                 packet_Start = find(Power_out_Norm>0.2);packet_Start = packet_Start(1)
% 
% 
%                 % if RX_State.ConstlationPlot
%                 %     figure();
%                 %     subplot(2,1,1);
%                 %     plot(Power_out_Norm);
%                 %     title('Signal Power')
%                 %     subplot(2,1,2);
%                 %     plot(Power_out_Norm_SNR)
%                 %     title('Auto Corr')
%                 % end
% 
%                 waveform = waveform(packet_Start:end);
% 
%                 Corr_out = zeros(1,500);
%                 for sample = 1:500
% 
%                     Corr_sample_out = 0;
%                     Corr_Sample_norm_power = 0;
% 
%                     for auto_ind = 1:48%(sample):(sample+1)
%                         sts_1 = waveform(sample+auto_ind-1:sample+auto_ind+14);
%                         sts_2 = waveform(sample+auto_ind+15:sample+auto_ind+30);
%                         Corr_sample_out = Corr_sample_out + sum(sts_1.*conj(sts_2));
%                           moving average filter 
%                        b1=(1/48)*ones(1,48) ; 
%                        Corr_sample_out = filter(b1,1,Corr_sample_out) ; 
% 
%                         Corr_sample_out = Corr_sample_out + sum(sts_1(auto_ind).*conj(sts_2(auto_ind)));
%                         Corr_Sample_norm_power = Corr_Sample_norm_power + sum(abs(sts_1))^2;%(sts_1.*conj(sts_1));
%                         moving average filter 
%                         b2=(1/48)*ones(1,48) ; 
%                         Corr_Sample_norm_power = filter(b2,1,Corr_Sample_norm_power) ; 
% 
%                     end
% 
%                     Corr_out(sample) = (abs(Corr_sample_out)^2)/Corr_Sample_norm_power;
%                     Corr_out(sample) = (Corr_sample_out)./Corr_Sample_norm_power
%                     Corr_out(sample) = sum(xcorr(sts_1,sts_2,'normalized'));
% 
%                 end
% 
% 
% 
%                 Corr_out = autocorr()
%                 if RX_State.ConstlationPlot
%                     figure();
%                     subplot(3,1,1);
%                     plot(1:length(waveform),abs(waveform))
%                     subplot(3,1,2);
%                     plot(Power_out_Norm);
%                     title('Signal Power')
%                     subplot(3,1,3);
%                     plot(Corr_out)
%                     title('Auto Corr')
% 
% 
%                 figure;
%                 scatter(STS_RMS, Peaks);
%                 title('Scatter Plot of Short Preamble RMS vs Peaks of Cross-Correlation');
%                 xlabel('RMS of Short Preamble');
%                 ylabel('Peaks of Cross-Correlation');
%                 end