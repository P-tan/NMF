
result_files = c("Release/TestNMF_MU.log", "Release/TestNMF_FastHALS.log")

result_file = result_files[1];
lapply(result_files, function(result_file)
{
    read.csv(result_file) -> df
    list(file = result_file, data = df[c("Time_msec", "NRV")])
}) ->  results

plot(results[[1]]$data, type="l", log="y")
lines(results[[2]]$data, col=2)
legend("topright",
       legend = c(lapply(results, function(result) result$file)),
       lty = 1,
       col = 1:2
       )
