/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "FastBoard.h"

#include <cassert>
#include <cctype>
#include <algorithm>
#include <array>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include "Utils.h"
#include "config.h"

using namespace Utils;

const int FastBoard::NUM_VERTICES;
const int FastBoard::NO_VERTEX;
const int FastBoard::PASS;
const int FastBoard::RESIGN;


int FastBoard::get_boardsize() const {
    return m_boardsize;
}

int FastBoard::get_vertex(int x, int y) const {
    assert(x >= 0 && x < BOARD_SIZE);
    assert(y >= 0 && y < BOARD_SIZE);
    assert(x >= 0 && x < m_boardsize);
    assert(y >= 0 && y < m_boardsize);

    int vertex = ((y + 1) * m_sidevertices) + (x + 1);

    assert(vertex >= 0 && vertex < m_numvertices);

    return vertex;
}

std::pair<int, int> FastBoard::get_xy(int vertex) const {
    //int vertex = ((y + 1) * (get_boardsize() + 2)) + (x + 1);
    int x = (vertex % m_sidevertices) - 1;
    int y = (vertex / m_sidevertices) - 1;

    assert(x >= 0 && x < m_boardsize);
    assert(y >= 0 && y < m_boardsize);
    assert(get_vertex(x, y) == vertex);

    return std::make_pair(x, y);
}

FastBoard::vertex_t FastBoard::get_state(int vertex) const {
    assert(vertex >= 0 && vertex < NUM_VERTICES);
    assert(vertex >= 0 && vertex < m_numvertices);

    return m_state[vertex];
}

///  终于找到了 落子函数
/// 这个fastboard只是实现棋盘状态改变部分,hash部分由其子类fullboard完成
void FastBoard::set_state(int vertex, FastBoard::vertex_t content) {
    assert(vertex >= 0 && vertex < NUM_VERTICES);
    assert(vertex >= 0 && vertex < m_numvertices);
    assert(content >= BLACK && content <= INVAL);

    m_state[vertex] = content;
    empty_cnt--;
}

FastBoard::vertex_t FastBoard::get_state(int x, int y) const {
    return get_state(get_vertex(x, y));
}

void FastBoard::set_state(int x, int y, FastBoard::vertex_t content) {
    set_state(get_vertex(x, y), content);
}

void FastBoard::reset_board(int size) {
    m_boardsize = size;
    m_sidevertices = size + 2;
    m_numvertices = m_sidevertices * m_sidevertices;
    m_tomove = BLACK;

    has_end = false;
    winner = EMPTY;
    empty_cnt = NUM_INTERSECTIONS;

    /// 下
    m_dirs[0] = -m_sidevertices;
    /// 右
    m_dirs[1] = +1;
    /// 上
    m_dirs[4] = +m_sidevertices;
    /// 左
    m_dirs[5] = -1;
    /// 右下
    m_dirs[2] = -m_sidevertices + 1;
    /// 右上
    m_dirs[3] = m_sidevertices + 1;
    /// 左上
    m_dirs[6] = m_sidevertices - 1;
    /// 左下
    m_dirs[7] = -m_sidevertices - 1;

    for (int i = 0; i < m_numvertices; i++) {
        m_state[i]     = INVAL;
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int vertex = get_vertex(i, j);
            m_state[vertex]           = EMPTY;
        }
    }

    assert(m_state[NO_VERTEX] == INVAL);
}

// Needed for scoring passed out games not in MC playouts
// 算输赢
// 总是站在黑方的角度来看
// 如果黑方五连 返回1
//    白方五连 返回-1
//    都没有   返回0
float FastBoard::end_score() const {
    assert(winner != INVAL);
    if (!has_end) {
        assert(winner == EMPTY);
        return 0.0;
    } else {
        if (winner == BLACK) {
            return 1.0;
        } else if (winner == WHITE) {
            return -1.0;
        } else {
            assert(winner == EMPTY);
            return 0.0;
        }
    }
//    auto white = calc_reach_color(WHITE);
//    auto black = calc_reach_color(BLACK);
//    return black - white - komi;
}

// 显示棋盘
void FastBoard::display_board(int lastmove) {
    int boardsize = get_boardsize();

    myprintf("\n   ");
    print_columns();
    for (int j = boardsize-1; j >= 0; j--) {
        myprintf("%2d", j+1);
        if (lastmove == get_vertex(0, j))
            myprintf("(");
        else
            myprintf(" ");
        for (int i = 0; i < boardsize; i++) {
            if (get_state(i,j) == WHITE) {
                myprintf("O");
            } else if (get_state(i,j) == BLACK)  {
                myprintf("X");
            } /*else if (starpoint(boardsize, i, j)) {
                myprintf("+");}*/
            else {
                myprintf(".");
            }
            if (lastmove == get_vertex(i, j)) myprintf(")");
            else if (i != boardsize-1 && lastmove == get_vertex(i, j)+1) myprintf("(");
            else myprintf(" ");
        }
        myprintf("%2d\n", j+1);
    }
    myprintf("   ");
    print_columns();
    myprintf("\n");
}

// 显示棋盘时画坐标刻度用
void FastBoard::print_columns() {
    for (int i = 0; i < get_boardsize(); i++) {
        if (i < 25) {
            myprintf("%c ", (('a' + i < 'i') ? 'a' + i : 'a' + i + 1));
        } else {
            myprintf("%c ", (('A' + (i - 25) < 'I') ? 'A' + (i - 25) : 'A' + (i - 25) + 1));
        }
    }
    myprintf("\n");
}

std::string FastBoard::move_to_text(int move) const {
    std::ostringstream result;

    int column = move % m_sidevertices;
    int row = move / m_sidevertices;

    column--;
    row--;

    assert(move == FastBoard::PASS
           || move == FastBoard::RESIGN
           || (row >= 0 && row < m_boardsize));
    assert(move == FastBoard::PASS
           || move == FastBoard::RESIGN
           || (column >= 0 && column < m_boardsize));

    if (move >= 0 && move <= m_numvertices) {
        result << static_cast<char>(column < 8 ? 'A' + column : 'A' + column + 1);
        result << (row + 1);
    } else if (move == FastBoard::PASS) {
        result << "pass";
    } else if (move == FastBoard::RESIGN) {
        result << "resign";
    } else {
        result << "error";
    }

    return result.str();
}

int FastBoard::text_to_move(std::string move) const {
    transform(cbegin(move), cend(move), begin(move), tolower);

    if (move == "pass") {
        return PASS;
    } else if (move == "resign") {
        return RESIGN;
    } else if (move.size() < 2 || !std::isalpha(move[0]) || !std::isdigit(move[1]) || move[0] == 'i') {
        return NO_VERTEX;
    }

    auto column = move[0] - 'a';
    if (move[0] > 'i') {
        --column;
    }

    int row;
    std::istringstream parsestream(move.substr(1));
    parsestream >> row;
    --row;

    if (row >= m_boardsize || column >= m_boardsize) {
        return NO_VERTEX;
    }

    return get_vertex(column, row);
}

std::string FastBoard::move_to_text_sgf(int move) const {
    std::ostringstream result;

    int column = move % m_sidevertices;
    int row = move / m_sidevertices;

    column--;
    row--;

    assert(move == FastBoard::PASS
           || move == FastBoard::RESIGN
           || (row >= 0 && row < m_boardsize));
    assert(move == FastBoard::PASS
           || move == FastBoard::RESIGN
           || (column >= 0 && column < m_boardsize));

    // SGF inverts rows
    row = m_boardsize - row - 1;

    if (move >= 0 && move <= m_numvertices) {
        if (column <= 25) {
            result << static_cast<char>('a' + column);
        } else {
            result << static_cast<char>('A' + column - 26);
        }
        if (row <= 25) {
            result << static_cast<char>('a' + row);
        } else {
            result << static_cast<char>('A' + row - 26);
        }
    } else if (move == FastBoard::PASS) {
        result << "tt";
    } else if (move == FastBoard::RESIGN) {
        result << "tt";
    } else {
        result << "error";
    }

    return result.str();
}


int FastBoard::get_to_move() const {
    return m_tomove;
}

bool FastBoard::black_to_move() const {
    return m_tomove == BLACK;
}

bool FastBoard::white_to_move() const {
    return m_tomove == WHITE;
}

void FastBoard::set_to_move(int tomove) {
    m_tomove = tomove;
}


bool FastBoard::is_forbidden(int vertex, vertex_t color) const {
    if (vertex_t(color) == WHITE) {
        return false;
    }
    /// myprintf("my color: %d\n", color);
    std::array<unsigned short, 4> m_row_type{OTHER_TYPE, OTHER_TYPE, OTHER_TYPE, OTHER_TYPE};
    for (int direction = 0; direction < 4; direction++) {
        int temp_vertex = vertex + m_dirs[direction];
        int mycolornum = 1;
        while (in_table(temp_vertex)) {
            if (get_state(temp_vertex) == color) {
                mycolornum++;
                temp_vertex += m_dirs[direction];
            } else {
                break;
            }
        }
        int left_blank = 0;
        int dist2v = temp_vertex + m_dirs[direction];
        if (in_table(temp_vertex)) {
            if (get_state(temp_vertex) == EMPTY) {
                left_blank += 1;
                if (in_table(dist2v)) {
                    // TODO 如果隔一个空是自己棋子呢
                    if (get_state(dist2v) == EMPTY) {
                        left_blank += 1;
                    }
                }
            }
        }
        int back_direction = direction + 4;
        temp_vertex = vertex + m_dirs[back_direction];
        while (in_table(temp_vertex)) {
            if (get_state(temp_vertex) == color) {
                mycolornum++;
                temp_vertex += m_dirs[back_direction];
            } else {
                break;
            }
        }
        int right_blank = 0;
        dist2v = temp_vertex + m_dirs[back_direction];
        if (in_table(temp_vertex)) {
            if (get_state(temp_vertex) == EMPTY) {
                right_blank += 1;
                if (in_table(dist2v)) {
                    // TODO 如果隔一个空是自己棋子呢
                    if (get_state(dist2v) == EMPTY) {
                        right_blank += 1;
                    }
                }
            }
        }
        /// myprintf("mycolornum: %d\n", mycolornum);
        if (mycolornum > NUM_IN_A_ROW) {
            m_row_type[direction] = TOOLONG;
        } else if (mycolornum == NUM_IN_A_ROW) {
            m_row_type[direction] = FIVE;
        } else if (mycolornum == NUM_IN_A_ROW - 1) {
            if (left_blank >= 1 && right_blank >= 1) {
                m_row_type[direction] = LIVE_FOUR;
            } else if (left_blank + right_blank == 1) {
                m_row_type[direction] = CONG_FOUR;
            }
        } else if (mycolornum == NUM_IN_A_ROW - 2) {
            if ((left_blank >= 1 && right_blank >= 2)
                 || (left_blank >= 2 && right_blank >= 1)) {
                m_row_type[direction] = LIVE_TREE;
            }
        }
    }
    bool has_toolong = false, has_five = false;
    int num_live_three = 0, num_four = 0;
    for (int direc = 0; direc < 4; direc++) {
        if (m_row_type[direc] == TOOLONG) {
            has_toolong = true;
        } else if (m_row_type[direc] == FIVE) {
            has_five = true;
        } else if (m_row_type[direc] == LIVE_FOUR || m_row_type[direc] == CONG_FOUR) {
            num_four += 1;
        } else if (m_row_type[direc] == LIVE_TREE) {
            num_live_three += 1;
        }
    }
    if (has_five) {
        return false;
    } else {
        if (has_toolong) {
            /// myprintf("shit1\n");
            return true;
        }
        if (num_four >= 2 || num_live_three >= 2) {
            /// myprintf("shit2\n");
            return true;
        }
    }
    return false;
}

void FastBoard::update_continue_info(int vertex, vertex_t content) {
    assert(content == BLACK || content == WHITE);
    for (int direction = 0; direction < 4; direction++) {
        int temp_vertex = vertex + m_dirs[direction];
        int mycolornum = 1;
        while (in_table(temp_vertex)) {
            if (get_state(temp_vertex) == content) {
                mycolornum++;
                temp_vertex += m_dirs[direction];
            } else {
                break;
            }
        }
        int back_direction = direction + 4;
        temp_vertex = vertex + m_dirs[back_direction];
        while (in_table(temp_vertex)) {
            if (get_state(temp_vertex) == content) {
                mycolornum++;
                temp_vertex += m_dirs[back_direction];
            } else {
                break;
            }
        }
        if (mycolornum > NUM_IN_A_ROW) {
            has_end = true;
            winner = content;
        } else if (mycolornum == NUM_IN_A_ROW) {
            has_end = true;
            winner = content;
        }
    }
    if (empty_cnt == 0 && !has_end) {
        has_end = true;
        winner = EMPTY;
    }
}

bool FastBoard::in_table(int vertex) const {
    int x = (vertex % m_sidevertices) - 1;
    int y = (vertex / m_sidevertices) - 1;
    return (x >= 0 && x < m_boardsize && y >= 0 && y < m_boardsize);
}

bool FastBoard::game_end() const {
    return has_end;
}
