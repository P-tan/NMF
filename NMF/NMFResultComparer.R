
result_files = c("Release/TestNMF_MU.log", "Release/TestNMF_FastHALS.log")

result_file = result_files[1];
lapply(result_files, function(result_file)
{
    read.csv(result_file) -> df
    list(file = result_file, data = df[c("Time_msec", "NRV")])
}) ->  results
