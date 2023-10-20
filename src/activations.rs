use std::f32::consts::E;

fn sigmoid_activation(z: f32) -> f32 {
    let z = z.max(-60.0).min(60.0);
    1.0 / (1.0 + E.powf(-5.0 * z))
}

fn tanh_activation(z: f32) -> f32 {
    let z = z.max(-60.0).min(60.0);
    z.tanh()
}

fn sin_activation(z: f32) -> f32 {
    let z = z.max(-60.0).min(60.0);
    z.sin()
}

fn gauss_activation(z: f32) -> f32 {
    let z = z.max(-3.4).min(3.4);
    (-5.0 * z * z).exp()
}

fn relu_activation(z: f32) -> f32 {
    if z > 0.0 {
        z
    } else {
        0.0
    }
}

fn elu_activation(z: f32) -> f32 {
    if z > 0.0 {
        z
    } else {
        E.powf(z) - 1.0
    }
}

fn lelu_activation(z: f32) -> f32 {
    let leaky = 0.005;
    if z > 0.0 {
        z
    } else {
        leaky * z
    }
}

fn selu_activation(z: f32) -> f32 {
    let lam = 1.0507009873554804934193349852946;
    let alpha = 1.6732632423543772848170429916717;
    if z > 0.0 {
        lam * z
    } else {
        lam * alpha * (E.powf(z) - 1.0)
    }
}

fn softplus_activation(z: f32) -> f32 {
    let z = z.max(-60.0).min(60.0);
    (1.0 + z.exp()).ln()
}

fn identity_activation(z: f32) -> f32 {
    z
}

fn clamped_activation(z: f32) -> f32 {
    z.max(-1.0).min(1.0)
}

fn inv_activation(z: f32) -> f32 {
    if z == 0.0 {
        0.0
    } else {
        1.0 / z
    }
}

fn log_activation(z: f32) -> f32 {
    let z = z.max(1e-7);
    z.ln()
}

fn exp_activation(z: f32) -> f32 {
    let z = z.max(-60.0).min(60.0);
    z.exp()
}

fn abs_activation(z: f32) -> f32 {
    z.abs()
}

fn hat_activation(z: f32) -> f32 {
    if z.abs() <= 1.0 {
        1.0 - z.abs()
    } else {
        0.0
    }
}

fn square_activation(z: f32) -> f32 {
    z * z
}

fn cube_activation(z: f32) -> f32 {
    z * z * z
}

pub const FUNCTIONS: [fn(f32) -> f32; 18] = [
    sigmoid_activation,
    tanh_activation,
    sin_activation,
    gauss_activation,
    relu_activation,
    elu_activation,
    lelu_activation,
    selu_activation,
    softplus_activation,
    identity_activation,
    clamped_activation,
    inv_activation,
    log_activation,
    exp_activation,
    abs_activation,
    hat_activation,
    square_activation,
    cube_activation,
];