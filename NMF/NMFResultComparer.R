library(magrittr)

result_files = paste0("x64/Release/", c("TestNMF_MU.log", "TestNMF_FastHALS.log"))

result_file = result_files[1];
lapply(result_files, function(result_file)
{
    read.csv(result_file) -> df
    list(file = result_file, data = df[-1, c("Time_msec", "NRV")])
}) ->  results


ylim = sapply(results, function(result) result$data$NRV) %>% range
plot(results[[1]]$data, type="l", ylim = ylim)
lines(results[[2]]$data, col=2)
legend("topright",
       legend = c(lapply(results, function(result) result$file)),
       lty = 1,
       col = 1:2
       )
