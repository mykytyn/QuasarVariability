import pstats
stat = pstats.Stats('stats')
stat.sort_stats('time')
stat.print_stats()
