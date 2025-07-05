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
#[macro_export]
macro_rules! impl_enum_from {
    ($from:ty => $to:ty { $($variant:ident),+ $(,)? }) => {
        impl From<$from> for $to {
            fn from(value: $from) -> Self {
                match value {
                    $(
                        <$from>::$variant => <$to>::$variant,
                    )+
                }
            }
        }
    };
}
