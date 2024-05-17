from methods.methods import OptimizationStep


def print_header():
    print(f'{"Iter":10} {"f":14} {"gradfnorm":10} {"alpha":10}')


def print_callback(i: int, step: OptimizationStep):
    alpha_text = f'{step.step_length:<10.5f}'
    print(f'{i:<10} {step.after_function_value:<14.5f} {step.after_gradient_norm:<10.5f} {alpha_text}')
