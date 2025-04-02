function PQprtMOV (MOV, ODG)
% Print MOV values (PEAQ Basic version)

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:47 $

N = length (MOV);
PQ_NMOV_B = 11;
PQ_NMOV_A = 5;

fprintf ('Model Output Variables:\n');
if (N == PQ_NMOV_B)
    fprintf ('   BandwidthRefB: %g\n', MOV(1));
    fprintf ('  BandwidthTestB: %g\n', MOV(2));
    fprintf ('      Total NMRB: %g\n', MOV(3));
    fprintf ('    WinModDiff1B: %g\n', MOV(4));
    fprintf ('            ADBB: %g\n', MOV(5));
    fprintf ('            EHSB: %g\n', MOV(6));
    fprintf ('    AvgModDiff1B: %g\n', MOV(7));
    fprintf ('    AvgModDiff2B: %g\n', MOV(8));
    fprintf ('   RmsNoiseLoudB: %g\n', MOV(9));
    fprintf ('           MFPDB: %g\n', MOV(10));
    fprintf ('  RelDistFramesB: %g\n', MOV(11));
elseif (N == NMOV_A)
    fprintf ('        RmsModDiffA: %g\n', MOV(1));
    fprintf ('  RmsNoiseLoudAsymA: %g\n', MOV(2));
    fprintf ('     Segmental NMRB: %g\n', MOV(3));
    fprintf ('               EHSB: %g\n', MOV(4));
    fprintf ('        AvgLinDistA: %g\n', MOV(5));
else
    error ('Invalid number of MOVs');
end

fprintf ('Objective Difference Grade: %.3f\n', ODG);

return;
