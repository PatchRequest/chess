import chess


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import numpy as np
from collections import namedtuple, deque
from DQN import DQN
from stockfish import Stockfish
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))

class chess_ai:
    def __init__(self):
        self.board = chess.Board()
        self.policy_net = DQN(8*8,1).to(device)
        self.target_net = DQN(8*8,1).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps_done = 0
        self.epsilon = 0.9

    def select_action(self):
        moves = list(self.board.legal_moves)
        new_moves = []
        for move in moves:
            new_moves.append(move.uci())
        moves = new_moves
        state = self.board_to_tensor(self.board)

        sample = random.random()
        eps_threshold = self.epsilon
        if sample > eps_threshold:
            with torch.no_grad():
                return moves[self.policy_net(state).max(1)[1].view(1, 1)]
        else:
            return random.choice(moves)
    
    def train(self):
        optimizer = optim.Adam(self.policy_net.parameters())
        memory = deque(maxlen=100000)
        
        episode_durations = []
        num_episodes = int(sys.argv[2])
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            self.board.reset()
            # generate random board
            # random number between 0 and 30
            number_of_moves = random.randint(0,60)
            for i in range(number_of_moves):
                moves = list(self.board.legal_moves)
                new_moves = []
                for move in moves:
                    new_moves.append(move.uci())
                moves = new_moves
                self.board.push_uci(random.choice(moves))

            state = self.board_to_tensor(self.board)
            for t in range(int(sys.argv[3])):
                # Select and perform an action
                action = self.select_action()
                reward = self.get_reward(self.board)
                temp_board = self.board.copy()
                temp_board.push_uci(action)
                if self.board.is_game_over():
                    next_state = None
                else:
                    next_state = self.board_to_tensor(temp_board)
                # Store the transition in memory
                memory.append(Transition(state, self.move_to_tensor(action), next_state, reward))
                # Move to the next state
                state = next_state
                # Perform one step of the optimization (on the target network)
                self.optimize_model(memory, optimizer)
                if self.board.is_game_over():
                    episode_durations.append(t + 1)
                    break

            # Update the target network, copying all weights and biases in DQN
            print("Episode: ", i_episode)
            if i_episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
           
            

    def get_reward(self,board):
        stockfish = Stockfish()
        stockfish.set_depth(5)
        stockfish.set_fen_position(board.fen())
        score = stockfish.get_evaluation()
        if score['type'] == 'mate':
            if score['value'] > 0:
                return torch.tensor([100], device=device)
            else:
                return torch.tensor([-100], device=device)
        else:
            return torch.tensor([score['value']/100], device=device)
       
        
    def optimize_model(self, memory, optimizer):
        if len(memory) < 1000:
            return
        transitions = random.sample(memory, 1000)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch)
        next_state_values = torch.zeros(1000, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * 0.99) + reward_batch
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    
  
    def move_to_tensor(self,move):
        # a move is a string of 4 characters
        # the first two characters are the square the piece is moving from
        # the last two characters are the square the piece is moving to
        # the first character is a letter from a to h
        # the second character is a number from 1 to 8
        # the last character is a letter from a to h
        # the last character is a number from 1 to 8
        matrix = torch.zeros(8,8)
        matrix[ord(move[0])-97][int(move[1])-1] = 1
        matrix[ord(move[2])-97][int(move[3])-1] = 1
        return matrix.view(1,64)
    
    def board_to_tensor(self,board):
        # 8x8 matrix
        # 0 = empty
        # 1 = white pawn
        # 2 = white knight
        # 3 = white bishop
        # 4 = white rook
        # 5 = white queen
        # 6 = white king
        # -1 = black pawn  
        # -2 = black knight
        # -3 = black bishop
        # -4 = black rook
        # -5 = black queen
        # -6 = black king
        tensor = torch.zeros(8,8)
        for i in range(8):
            for j in range(8):
                piece = board.piece_at(chess.square(i,j))
                if piece == None:
                    tensor[i][j] = 0
                else:
                    if piece.color == chess.WHITE:
                        if piece.piece_type == chess.PAWN:
                            tensor[i][j] = 1
                        elif piece.piece_type == chess.KNIGHT:
                            tensor[i][j] = 2
                        elif piece.piece_type == chess.BISHOP:
                            tensor[i][j] = 3
                        elif piece.piece_type == chess.ROOK:
                            tensor[i][j] = 4
                        elif piece.piece_type == chess.QUEEN:
                            tensor[i][j] = 5
                        elif piece.piece_type == chess.KING:
                            tensor[i][j] = 6
                    else:
                        if piece.piece_type == chess.PAWN:
                            tensor[i][j] = -1
                        elif piece.piece_type == chess.KNIGHT:
                            tensor[i][j] = -2
                        elif piece.piece_type == chess.BISHOP:
                            tensor[i][j] = -3
                        elif piece.piece_type == chess.ROOK:
                            tensor[i][j] = -4
                        elif piece.piece_type == chess.QUEEN:
                            tensor[i][j] = -5
                        elif piece.piece_type == chess.KING:
                            tensor[i][j] = -6

        return tensor.view(1,8*8)

        
        

myAi = chess_ai()
# mode is argument passed to the program
# train to train the model
# play to play against the model
mode = sys.argv[1]
if mode == "train":
    # load the model if it exists
    try:
        myAi.policy_net.load_state_dict(torch.load('policy_net.pth'))
        myAi.target_net.load_state_dict(torch.load('target_net.pth'))
        myAi.policy_net.eval()
        myAi.target_net.eval()
    except:
        pass
    myAi.train()
    torch.save(myAi.policy_net.state_dict(), 'policy_net.pth')
    torch.save(myAi.target_net.state_dict(), 'target_net.pth')
else:
    # load the model
    myAi.policy_net.load_state_dict(torch.load('policy_net.pth'))
    myAi.target_net.load_state_dict(torch.load('target_net.pth'))
    myAi.policy_net.eval()
    myAi.target_net.eval()
    # make a move then the network makes a move until the game is over
    while not myAi.board.is_game_over():
        # make a move
        try:
            print("your move")
            move = input()
            if move == "quit":
                break
            myAi.board.push_uci(move)
            # make the network make a move
            myAi.board.push_uci(myAi.select_action())
            print(myAi.board)
        except:
            print("invalid move")
            continue
        
        


# check what the network would do as a first move and then make that move
myAi.board.reset()
myAi.board.push_uci(myAi.select_action())
print(myAi.board)



