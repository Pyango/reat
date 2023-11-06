use chess::{Board, BoardStatus, CastleRights, ChessMove, Color, File, Game, MoveGen, Rank, Square};
use crate::genome::Genome;

unsafe fn board_to_input(board: &Board) -> [[f32; 8]; 8] {
    let mut matrix = [[0f32; 8]; 8];

    for rank in 0..8 {
        for file in 0..8 {
            let square = chess::Square::new(rank * 8 + file);
            match board.piece_on(square) {
                Some(piece) => {
                    let piece_value = match piece {
                        chess::Piece::Pawn => 1.0,
                        chess::Piece::Knight => 2.0,
                        chess::Piece::Bishop => 3.0,
                        chess::Piece::Rook => 4.0,
                        chess::Piece::Queen => 5.0,
                        chess::Piece::King => 6.0,
                    };
                    matrix[rank as usize][file as usize] = piece_value;
                    if board.color_on(square) == Some(chess::Color::White) {
                        matrix[rank as usize][file as usize] *= -1.0;
                    }
                }
                None => {}
            }
        }
    }

    // Castling rights
    if board.castle_rights(Color::White) == CastleRights::Both {
        matrix[7][7] = 7.0;  // White king-side rook's position
        matrix[7][0] = 8.0;  // White queen-side rook's position
    }
    if board.castle_rights(Color::White) == CastleRights::KingSide {
        matrix[7][7] = 7.0;  // White king-side rook's position
    }
    if board.castle_rights(Color::White) == CastleRights::QueenSide {
        matrix[7][0] = 8.0;  // White queen-side rook's position
    }
    if board.castle_rights(Color::Black) == CastleRights::Both {
        matrix[0][7] = -7.0;  // White king-side rook's position
        matrix[0][0] = -8.0;  // White queen-side rook's position
    }
    if board.castle_rights(Color::Black) == CastleRights::KingSide {
        matrix[0][7] = -7.0;  // White king-side rook's position
    }
    if board.castle_rights(Color::Black) == CastleRights::QueenSide {
        matrix[0][0] = -8.0;  // White queen-side rook's position
    }

    // En passant
    if let Some(square) = board.en_passant() {
        let rank = if board.side_to_move() == chess::Color::White { square.get_rank().to_index() + 1 } else { square.get_rank().to_index() - 1 };
        let file = square.get_file().to_index();
        matrix[rank][file] = if board.side_to_move() == chess::Color::White { 9.0 } else { -9.0 };
    }
    if board.side_to_move() == chess::Color::Black {
        for rank in 0..8 {
            for file in 0..8 {
                matrix[rank][file] *= -1.0;
            }
        }
    }


    matrix
}

fn matrix_to_move(matrix: [[f32; 8]; 8]) -> ChessMove {
    let mut min_value = std::f32::MAX;
    let mut max_value = std::f32::MIN;

    let mut source = None;
    let mut destination = None;

    for rank in 0..8 {
        for file in 0..8 {
            let value = matrix[rank][file];
            if value != 0.0 {  // Considering non-zero values
                if value < min_value as f32 {
                    min_value = value;
                    source = Some(Square::make_square(Rank::from_index(7 - rank), File::from_index(file)));
                }
                if value > max_value as f32 {
                    max_value = value;
                    destination = Some(Square::make_square(Rank::from_index(7 - rank), File::from_index(file)));
                }
            }
        }
    }

    if source.is_none() || destination.is_none() {
        panic!("Invalid matrix!");
    }

    ChessMove::new(source.unwrap(), destination.unwrap(), None)
}

fn is_move_legal(board: &Board, m: ChessMove) -> bool {
    let movegen = MoveGen::new_legal(&board);

    for legal_move in movegen {
        if legal_move == m {
            return true;
        }
    }
    false
}

fn flatten(input: [[f32; 8]; 8]) -> Vec<f32> {
    let mut flat = Vec::new();
    for row in &input {
        for &value in row {
            flat.push(value);
        }
    }
    flat
}

fn unflatten(input: Vec<f32>) -> [[f32; 8]; 8] {
    let mut result: [[f32; 8]; 8] = Default::default();

    for i in 0..8 {
        for j in 0..8 {
            let index = i * 8 + j;
            result[i][j] = input[index];
        }
    }

    result
}

pub unsafe fn play_game(genome: &Genome) -> (f32, Game) {
    let mut game = Game::new();

    loop {
        let mut board = game.current_position();
        // Check if the game is in checkmate or stalemate
        if board.status() == BoardStatus::Checkmate {
            println!("Checkmate");
            *genome.fitness.borrow_mut() -= 50.0;
            break;
        } else if board.status() == BoardStatus::Stalemate {
            println!("Stalemate");
            *genome.fitness.borrow_mut() -= 25.0;
            break;
        }

        let board_input = board_to_input(&board);
        let flattened_board = flatten(board_input);
        let move_prediction = genome.activate(&flattened_board);
        let move_matrix = unflatten(move_prediction);
        let m = matrix_to_move(move_matrix);

        if is_move_legal(&board, m) {
            game.make_move(m);
            board = game.current_position();

            if game.can_declare_draw() {
                println!("can_declare_draw");
                break;
            }

            // Award 1 point for a valid move.
            // println!("VALID MOVE!!");
            *genome.fitness.borrow_mut() += 1.0;
        } else {
            // println!("Invalid Move XD, {}", m);
            // Deduct 50 points for an invalid move and stop the game.
            *genome.fitness.borrow_mut() -= 50.0;
            break;
        }
    }
    // println!("Side to move {:?}", game.side_to_move());
    (*genome.fitness.borrow(), game)
}
