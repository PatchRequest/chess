import chess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
from stockfish import Stockfish
import sys

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    agent = chess_white_ai()

    try:
        agent.load_model()
    except:
        pass

    if sys.argv[1] == 'train':
        agent.train()
        agent.save_model()
    else:
        agent.play()
        pass
    agent.board.reset()
    print(agent.select_action())

    pass

class chess_neural_net(nn.Module):
    def __init__(self):
        super(chess_neural_net, self).__init__()
        self.net = nn.Sequential(
            # input is 1x1x64
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1)
        )


    def forward(self, x):
        x = x.view(-1, 384)
        return self.net(x)
    

    
class chess_white_ai:
    def __init__(self):
        self.board = chess.Board()
        self.policy_net = chess_neural_net().to(device)
        self.target_net = chess_neural_net().to(device) 
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = 0.7
    
    def play(self):
        print("play")
        self.board.reset()
        self.epsilon = 1
        # print board
        print(self.board)
        while not self.board.is_game_over():
            move = self.select_action()
            print(move)
            self.board.push(chess.Move.from_uci(move))
            print(self.board)
            inputMove = input("Enter move: ")
            while not self.board.is_legal(chess.Move.from_uci(inputMove)):
                inputMove = input("Enter move: ")
            self.board.push(chess.Move.from_uci(inputMove))
            print(self.board)

    def select_action(self,board=None):
        if board == None:
            board = self.board
        available_moves = list(move.uci() for move in  board.legal_moves)
        if len(available_moves) == 0:
            return None
        board_move_tensors = []
        for move in available_moves:
            board_move_tensors.append(self.combine_board_move(board,move))

        board_move_tensors = torch.stack(board_move_tensors)
        if random.random() < self.epsilon:
            return random.choice(available_moves)
        else:
            return available_moves[torch.argmax(self.policy_net(board_move_tensors)).item()]
      
        
    def get_reward(self,board=None):
        if board == None:
            board = self.board
        stockfish = Stockfish()
        stockfish.set_fen_position(board.fen())
        eval_stock = stockfish.get_evaluation()
        if eval_stock['type'] == 'mate':
            if eval_stock['value'] > 0:
                return 6000
            elif eval_stock['value'] == 0:
                if board.result() == "1-0":
                    return 9000
                elif board.result() == "0-1":
                    return -9000
                else:
                    return 0
            elif eval_stock['value'] < 0:
                return -6000
        else:
            return int(eval_stock['value'])
        
    def combine_board_move(self,board,move):
        boardTensor = self.board_to_tensor(board)
        moveTensor = self.move_to_tensor(move)
        # board is 1x320
        # move is 1x64
        # combined is 1x384
        return torch.cat((boardTensor,moveTensor),1)
        
    def combine_tensorBoard_move(self,tensorBoard,move):
        tensorBoard = tensorBoard
        moveTensor = self.move_to_tensor(move)
        # board is 1x320
        # move is 1x64
        # combined is 1x384
        return torch.cat((tensorBoard,moveTensor),1)
        

    
    def eval_move(self,move):
        return self.policy_net(self.combine_board_move(self.board,move))
    
    def board_to_tensor(self,board=None):
        if board == None:
            board = self.board
        tensor = torch.zeros(8,8,5)
        # first layer is board repesentet by positive for white and negative for black
        # second layer matrix of all attacked fields by white
        # third layer matrix of all attacked fields by black
        # forth layer is filled with how many times the position was already visited
        # fifth layer marked available castling moves with 1 for white and -1 for black

        for i in range(8):
            for j in range(8):
                piece = board.piece_at(i*8+j)
                if piece != None:
                    if piece.color == chess.WHITE:
                        tensor[i][j][0] = piece.piece_type
                    else:
                        tensor[i][j][0] = -piece.piece_type
                    if board.is_attacked_by(chess.WHITE,i*8+j):
                        tensor[i][j][1] = 1
                    if board.is_attacked_by(chess.BLACK,i*8+j):
                        tensor[i][j][2] = 1
                    if board.is_repetition(2):
                        tensor[i][j][3] = 2
                    if board.is_repetition(3):
                        tensor[i][j][3] = 3
        rights = board.castling_rights
        if rights & chess.BB_H1:
            tensor[7][7][4] = 1
        if rights & chess.BB_A1:
            tensor[7][0][4] = 1
        if rights & chess.BB_H8:
            tensor[0][7][4] = -1
        if rights & chess.BB_A8:
            tensor[0][0][4] = -1

        return tensor.view(1,320)
    
    def move_to_tensor(self,move):
        tensor = torch.zeros(8,8)
        tensor[ord(move[0])-97][int(move[1])-1] = -1
        tensor[ord(move[2])-97][int(move[3])-1] = 1
        return tensor.view(1,64)

        
    def train(self,episode_count=1000):
        optimizer = optim.Adam(self.policy_net.parameters())
        memory = deque(maxlen=10000)
        episode_won = 0
        episode_lost = 0
        episode_draw = 0
        episode_count = 1002
        difference_avg = 0
        difference_count = 0
        # open train.log in append mode
        with open("train.log", "a") as f:

            for i in range(episode_count):
                length = 0
                self.board.reset()
                while not self.board.is_game_over():
                    length += 1
                    state = self.board_to_tensor()
                    action = self.select_action()
                    if action == None:
                        break
                    self.board.push(chess.Move.from_uci(action))
                    next_state = self.board_to_tensor()
                    reward = self.get_reward()
                    memory.append((state, action, next_state, reward))
                    prediction = self.eval_move(action)
                    self.optimize_model(memory,optimizer)
                    # use stockfish to make a counter move
                    if not self.board.is_game_over():
                        if random.random() < .3:
                            stockfish = Stockfish()
                            stockfish.set_fen_position(self.board.fen())
                            move = stockfish.get_best_move()
                            self.board.push(chess.Move.from_uci(move))
                        else:
                            move = random.choice(list(self.board.legal_moves)).uci()
                            self.board.push(chess.Move.from_uci(move))
                    difference = prediction.item() - reward
                    # calculate average difference
                    difference_count += 1
                    difference_avg += 1/difference_count * (difference - difference_avg)
                    
                    #print(i, " ", difference_avg)
                    #print("episode: ",i," reward: ",reward," length: ",length, "Prediction: ",prediction.item()," Move: ",move)
                    print(f"Episode: {i} \t Reward: {reward} \t Length: {length} \t Prediction: {prediction.item()} \t Move: {move} \t AvgDiff: {difference_avg}")
                    f.write(str(difference_avg)+"\n")
                    if i % 10 == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                    if i % 100 == 0:
                        self.save_model()
                
                if self.board.result() == "1-0":
                    episode_won += 1
                elif self.board.result() == "0-1":
                    episode_lost += 1
                else:
                    episode_draw += 1
                print(f"Won by checkmate: {episode_won} Lost by checkmate: {episode_lost} Draw: {episode_draw}")

    def save_model(self):
        torch.save(self.policy_net.state_dict(), "policy_net.pth")
        torch.save(self.target_net.state_dict(), 'target_net.pth')
    def load_model(self):
        self.policy_net.load_state_dict(torch.load("policy_net.pth"))
        self.target_net.load_state_dict(torch.load("target_net.pth"))

    def optimize_model(self,memory,optimizer):
        if len(memory) > 32:
            batch = random.sample(memory, 32)
            for state, action, next_state, reward in batch:
                target = reward
                target = torch.tensor([[target]], dtype=torch.float)
                
                
                board_move = self.combine_tensorBoard_move(state,action)
                prediction = self.policy_net(board_move)
                loss = F.smooth_l1_loss(prediction, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    
if __name__ == "__main__":
    main()