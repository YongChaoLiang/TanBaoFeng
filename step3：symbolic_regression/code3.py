"""
进行画图
"""

import pickle
import graphviz

if __name__ == '__main__':
    with open('./model/model8/gp_model.pkl', 'rb') as f:
        gp = pickle.load(f)

    for program in gp:
        print(program)
        print(program.raw_fitness_)
        print(program.fitness_)
        print(program.depth_)
        print(program.length_)
        #
        dot_data = program.export_graphviz()
        graph = graphviz.Source(dot_data)
        graph.view(directory='./model/model8')