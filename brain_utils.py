

class ConvergenceLogger():
    def __init__(self):
        #self.support = set()
        #self.winners_hist = []
        #self.support_sizes = []
        #self.num_new_winners = []
        self.step = 0
        self.winners = {} #winners at each time step
        self.new_winners = {}
        self.percent_new_winners = {}

    def update(self, timestep, winners):

        #which neurons fired at each time step!
        self.winners[timestep] = winners

        #update time step
        self.step = timestep

        return
        
    



    def write(self):
        return

        

