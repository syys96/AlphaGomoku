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

#include "GameState.h"
#include "Network.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <iterator>
#include <memory>
#include <string>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "KoState.h"
#include "UCTSearch.h"

void GameState::init_game(int size) {
    KoState::init_game(size);

    game_history.clear();
    game_history.emplace_back(std::make_shared<KoState>(*this));

    m_timecontrol.reset_clocks();

    m_resigned = FastBoard::EMPTY;
}

void GameState::reset_game() {
    KoState::reset_game();

    game_history.clear();
    game_history.emplace_back(std::make_shared<KoState>(*this));

    m_timecontrol.reset_clocks();

    m_resigned = FastBoard::EMPTY;
}

bool GameState::forward_move() {
    if (game_history.size() > m_movenum + 1) {
        m_movenum++;
        *(static_cast<KoState*>(this)) = *game_history[m_movenum];
        return true;
    } else {
        return false;
    }
}

bool GameState::undo_move() {
    if (m_movenum > 0) {
        m_movenum--;

        // this is not so nice, but it should work
        *(static_cast<KoState*>(this)) = *game_history[m_movenum];

        // This also restores hashes as they're part of state
        return true;
    } else {
        return false;
    }
}

void GameState::rewind() {
    *(static_cast<KoState*>(this)) = *game_history[0];
    m_movenum = 0;
}

void GameState::play_move(int vertex) {
    play_move(get_to_move(), vertex);
}

void GameState::play_move(int color, int vertex) {
    if (vertex == FastBoard::RESIGN) {
        m_resigned = color;
    } else {
        KoState::play_move(color, vertex);
    }

    // cut off any leftover moves from navigating
    game_history.resize(m_movenum);
    game_history.emplace_back(std::make_shared<KoState>(*this));
}

bool GameState::play_textmove(std::string color, const std::string& vertex) {
    int who;
    transform(cbegin(color), cend(color), begin(color), tolower);
    if (color == "w" || color == "white") {
        who = FullBoard::WHITE;
    } else if (color == "b" || color == "black") {
        who = FullBoard::BLACK;
    } else {
        return false;
    }

    const auto move = board.text_to_move(vertex);
    if (move == FastBoard::NO_VERTEX ||
        (move != FastBoard::PASS && move != FastBoard::RESIGN && board.get_state(move) != FastBoard::EMPTY)) {
        return false;
    }

    set_to_move(who);
    play_move(move);

    return true;
}

void GameState::stop_clock(int color) {
    m_timecontrol.stop(color);
}

void GameState::start_clock(int color) {
    m_timecontrol.start(color);
}

void GameState::display_state() {
    FastState::display_state();

    m_timecontrol.display_times();
}

int GameState::who_resigned() const {
    return m_resigned;
}

bool GameState::has_resigned() const {
    return m_resigned != FastBoard::EMPTY;
}

const TimeControl& GameState::get_timecontrol() const {
    return m_timecontrol;
}

void GameState::set_timecontrol(const TimeControl& timecontrol) {
    m_timecontrol = timecontrol;
}

void GameState::set_timecontrol(int maintime, int byotime,
                                int byostones, int byoperiods) {
    TimeControl timecontrol(maintime, byotime,
                            byostones, byoperiods);

    m_timecontrol = timecontrol;
}

void GameState::adjust_time(int color, int time, int stones) {
    m_timecontrol.adjust_time(color, time, stones);
}

void GameState::anchor_game_history() {
    // handicap moves don't count in game history
    m_movenum = 0;
    game_history.clear();
    game_history.emplace_back(std::make_shared<KoState>(*this));
}

const FullBoard& GameState::get_past_board(int moves_ago) const {
    assert(moves_ago >= 0 && (unsigned)moves_ago <= m_movenum);
    assert(m_movenum + 1 <= game_history.size());
    return game_history[m_movenum - moves_ago]->board;
}

const std::vector<std::shared_ptr<const KoState>>& GameState::get_game_history() const {
    return game_history;
}

bool GameState::has_end() const {
    return board.game_end();
}
