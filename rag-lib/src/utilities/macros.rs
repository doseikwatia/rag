#[macro_export]
macro_rules! dprintln {
    ($($x:tt)*) => {
        
        #[cfg(debug_assertions)]
        {
            use colored::Colorize;
            let text = format!($($x)*).green();
            println!("{}",text);
        }
    }
}
